"""Summarize equal-timing recovery screening and the existing AirSim pair."""
import argparse
import csv
from pathlib import Path
import re

import numpy as np

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .probe_main18_matched_recovery import BUNDLE, MODELS


def architecture(name):
    return tuple(map(int, re.search(r'_h(\d+)_w(\d+)_', name).groups()))


def label(name):
    return '%d×%d' % architecture(name)


def audit_range(replay_inputs):
    norm_path = BUNDLE/'shared/normalization.json'
    norm = read_json(norm_path)
    scale = 16.  # Reported by the original 5237 startup for all 18 loaded models.
    rows = []
    for sample in read_json(replay_inputs)['samples']:
        x = np.asarray(sample['raw237'], dtype=np.float32)
        z = (x-np.asarray(norm['x_mean'], dtype=np.float32))/np.asarray(norm['x_std'], dtype=np.float32)*scale
        over = np.flatnonzero((z >= 128.) | (z <= -129.))
        rows.append(dict(case=sample['case'], sample_s=sample['sample_s'], over_channels=over.tolist(),
                         minimum_quantizer_headroom=float(np.min(np.minimum(128-z, z+129)))))
    return dict(scale=scale, normalization_sha256=sha256(norm_path), source=str(replay_inputs.resolve()),
                predicted_rejected=sum(bool(r['over_channels']) for r in rows), samples=rows)


def analyze(offline, airsim, output, replay_inputs):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    a, b = [read_json(path/'summary.json') for path in (offline, airsim)]
    if not a['complete'] or not b['complete'] or not all(p['matched'] for p in a['pairs']+b['pairs']):
        raise ValueError('complete, matched inputs required')
    names = sorted(a['pairs'][0]['metrics'], key=architecture)
    if len(names) != 18:
        raise ValueError('expected all 18 configurations')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    def save(fig, stem):
        for extension in ('png', 'svg'):
            fig.savefig(output/f'{stem}.{extension}', dpi=200, bbox_inches='tight')
        plt.close(fig)

    rows, screen, matrix = [], [], []
    for pair in a['pairs']:
        metrics = pair['metrics']
        group = []
        for name in names:
            m = metrics[name]
            fine = m['refined_step_metrics']
            group.append(dict(case=pair['case'], model=name, depth=architecture(name)[0],
                              width=architecture(name)[1], tracking_rmse_m=m['position_rmse_m'],
                              recovery_s=m['recovery_s'], recovered=m['tracking_recovered'],
                              min_frame_margin_m=m['minimum_frame_margin_m'],
                              refined_frame_margin_m=fine['minimum_frame_margin_m'],
                              refined_rmse_m=fine['position_rmse_m'], refined_recovered=fine['tracking_recovered']))
        rows.extend(group)
        margins = [r['min_frame_margin_m'] for r in group]
        matrix.append(margins)
        screen.append(dict(case=pair['case'], min_frame_margin_m=min(margins), max_frame_margin_m=max(margins),
                           rmse_min_m=min(r['tracking_rmse_m'] for r in group),
                           rmse_max_m=max(r['tracking_rmse_m'] for r in group),
                           positive_margin_and_recovery=sum(r['min_frame_margin_m'] > 0 and r['recovered'] for r in group),
                           refined_positive_margin_and_recovery=sum(
                               r['refined_frame_margin_m'] > 0 and r['refined_recovered'] for r in group),
                           max_step_margin_difference_m=max(abs(r['refined_frame_margin_m']-r['min_frame_margin_m'])
                                                            for r in group)))
    with (output/'offline_models.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    values = np.array(matrix).T*100
    fig, ax = plt.subplots(figsize=(7.1, 8.2), layout='constrained')
    heatmap = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(vmin=-6, vcenter=0, vmax=30), aspect='auto')
    ax.set_xticks(range(len(screen)), [p['case'][2:] for p in screen])
    ax.set_yticks(range(len(names)), [label(n) for n in names])
    ax.set_xlabel('Initial lateral speed error magnitude (m/s)')
    ax.set_ylabel('Model depth × width')
    ax.set_title('Equal-timing screen across 18 FP32 models', loc='left', weight='bold', pad=12)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:+.1f}', ha='center', va='center',
                    color='white' if abs(value) > (3.5 if value < 0 else 19) else '#222222', fontsize=9)
    fig.colorbar(heatmap, ax=ax, shrink=.7, label='Minimum sampled frame margin (cm)')
    fig.text(.015, -.025, 'Reduced dynamics. All models sample every 50 ms and apply after 20 ms.', fontsize=9)
    save(fig, 'main18_recovery_heatmap')

    severe = b['pairs'][-1]['case']
    severe_metrics = b['pairs'][-1]['metrics']
    traces = [read_json(airsim/severe/name/'trace.json') for name in MODELS]
    failures = [severe_metrics[name]['physical']['failure'] for name in MODELS]
    common_end = min(*(t[-1]['t'] for t in traces), *(f['t'] for f in failures if f))
    grid = np.linspace(0, common_end, 501)
    common_rmse = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout='constrained')
    for i, (name, trace, failure, color, style) in enumerate(zip(
            MODELS, traces, failures, ('#0072B2', '#D55E00'), ('-', '--'))):
        t, e = (np.asarray([r[key] for r in trace]) for key in ('t', 'position_error_m'))
        interp = np.interp(grid, t, e)
        common_rmse[name] = float(np.sqrt(np.trapz(interp**2, grid)/common_end))
        axes[1].plot(t, e, style, color=color, linewidth=2, label=label(name))
        if failure:
            at = failure['t']
            axes[1].scatter(at, np.interp(at, t, e), color=color, marker='x', s=55, zorder=3)
        metrics = [p['metrics'][name]['position_rmse_m'] for p in b['pairs'][:-1]]
        axes[0].plot(range(len(metrics)), metrics, style, marker=('o', 'x')[i],
                     color=color, linewidth=1.8, label=label(name))
    axes[0].set_xticks(range(len(b['pairs'])-1), [p['case'][2:] for p in b['pairs'][:-1]])
    axes[0].set(xlabel='Initial lateral speed error (m/s)', ylabel='Position RMSE (m)')
    axes[0].set_title('a  Both models finish', loc='left', weight='bold')
    axes[0].legend(frameon=False, title='Depth × width')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Position error (m)')
    axes[1].set_title('b  Both contact the same gate', loc='left', weight='bold')
    axes[1].axvspan(0, common_end, color='0.9', alpha=.6, zorder=0)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='0.9')
    fig.text(.50, -.055, 'Gray = common interval for RMSE. × = first observed gate contact.', fontsize=9)
    save(fig, 'main18_airsim_matched_pair')

    physical = [dict(case=p['case'], models=[
        dict(model=name, reported_rmse_m=m['position_rmse_m'], recovered=m['physical']['recovery_passed'],
             failure=m['physical']['failure'], minimum_route_h_m=m['physical']['minimum_h'])
        for name, m in p['metrics'].items()], maximum_drift_s=p['maximum_matched_timing_difference_s'])
        for p in b['pairs']]
    atomic_json(output/'summary.json', dict(
        sources={key: dict(directory=str(path.resolve()), summary_sha256=sha256(path/'summary.json'))
                 for key, path in (('offline', offline), ('airsim', airsim))},
        simplified_screen=screen, airsim=physical, severe_common_end_s=common_end,
        severe_common_time_weighted_rmse_m=common_rmse, severe_trace_sha256={
            name: sha256(airsim/severe/name/'trace.json') for name in MODELS},
        local_range_estimate=audit_range(replay_inputs) if replay_inputs else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', type=Path, required=True)
    parser.add_argument('--airsim', type=Path, required=True)
    parser.add_argument('--replay-inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.offline, args.airsim, args.output, args.replay_inputs)
