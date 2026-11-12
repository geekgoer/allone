"""Replot Route B meeting figures from the original experiment JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
OUTPUT = ROOT/'docs/figures/route_b_20261110'
BLUE, ORANGE, GREEN, PURPLE = '#0072B2', '#B84B00', '#007A59', '#834796'
COLORS = (BLUE, ORANGE, GREEN, PURPLE)
MARKERS = ('o', 's', '^', 'D')


def load(path):
    return json.loads((RESULTS/path).read_text(encoding='utf-8'))


def save(fig, output, name):
    fig.savefig(output/f'{name}.png', dpi=185, facecolor='white')
    fig.savefig(output/f'{name}.svg', facecolor='white')
    plt.close(fig)


def action_figure(output):
    data = load('route_b_action_replay_20261109_v1/summary.json')
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    columns = [data['action_columns'].index(c) for c in ('T', 'c', 'schedule', 'T+c', 'all')]
    counts = np.asarray(data['counts'])[:, columns]
    totals = np.asarray(data['denominators'])
    values = counts/totals[:, None]
    a.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    for (i, j), value in np.ndenumerate(values):
        a.text(j, i, f'{counts[i,j]}/{totals[i]}', ha='center', va='center',
               color='white' if value > .64 else '#162739')
    a.set(xticks=range(5), xticklabels=['T only', 'c only', 'Order only', 'T + c', 'All'],
          yticks=range(4), yticklabels=data['strata'], xlabel='Allowed actions',
          ylabel='Generated scenario stratum', title='A  |  Which actions can fix the miss?')
    styles = [('fixed_order', 'Fixed order', '#555555', 'x', ':'),
              ('local', 'Local search', PURPLE, 'v', '--'),
              ('mode_only', 'Model only', GREEN, '^', '--'),
              ('abc', 'ABC', BLUE, 'o', '-'),
              ('guarded_abc', 'ABC + dispatch guard', ORANGE, 's', '-.')]
    for key, label, color, marker, line in styles:
        b.plot(range(1, 6), data['curves'][key], label=label, color=color, marker=marker,
               linestyle=line, linewidth=1.8, markerfacecolor='white')
    b.axhline(data['oracle_feasible'], color='#777777', linewidth=.8, linestyle=':')
    b.text(1.08, 133, 'Finite oracle: 131', color='#444444')
    b.set(xticks=range(1, 6), ylim=(0, 143), xlabel='Full checks allowed (including baseline)',
          ylabel='Scenarios repaired', title='B  |  Repairs versus decision budget')
    b.legend(loc='upper right', bbox_to_anchor=(1, .83), frameon=False, fontsize=8)
    fig.suptitle('Controlled scheduling replay  |  160 scenarios, 147 initial misses')
    save(fig, output, '01_abc_replay')


def model_figure(output):
    cost = load('route_b_action_replay_20261109_v1/protocol.json')['service_ms']
    trials = load('main18_full_routes_20261109_201022/summary.json')['trials']
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for y, (model, label, color) in enumerate([('11', '4×1024', BLUE), ('6', '33×1024', ORANGE)]):
        a.barh(y, cost[model], color=color, height=.48)
        a.text(cost[model]+.22, y, f'{cost[model]:.3f}', va='center')
    a.set(yticks=[0, 1], yticklabels=['4×1024', '33×1024'], xlim=(0, 13.7),
          xlabel='Board service P99 (ms)', title='A  |  Real DPU, two resident INT8 models')
    models = ['h2_w128', 'h4_w1024', 'h6_w1024', 'h9_w512']
    routes = ['smooth', 'turns', 'dense_turns']
    for j, (model, label) in enumerate(zip(models, ['2×128', '4×1024', '6×1024', '9×512'])):
        values = []
        for route in routes:
            rows = [r for r in trials if r['route'] == route]
            reference = next(r for r in rows if '_h4_w1024_' in r['model'])
            sample = next(r for r in rows if f'_{model}_' in r['model'])
            values.append(1000*(sample['minimum_crossing_margin_m']-reference['minimum_crossing_margin_m']))
        b.scatter(np.arange(3)+(j-1.5)*.13, values, label=label, color=COLORS[j], marker=MARKERS[j], s=44)
    b.axhline(0, color='#666666', linewidth=.9, linestyle=':')
    b.set(xticks=range(3), xticklabels=['Smooth', 'Turns', 'Dense turns'],
          ylabel='Minimum gate clearance vs 4×1024 (mm)',
          title='B  |  AirSim FP32, equal timing, 12/12 passed')
    b.legend(frameon=False, ncol=2, fontsize=8, loc='upper right', bbox_to_anchor=(1, .76))
    fig.suptitle('Model choice  |  Measured cost and measured control utility')
    save(fig, output, '02_model_cost_and_utility')


def phase_figure(output):
    natural = load('ip37_held_gap_natural_20261109_220202/summary.json')['pairs']
    rebuilt = load('ip37_held_gap_airsim_20261109_220031/summary.json')['pairs']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.65), sharey=True, layout='constrained')
    cases = ('T50_early', 'T50_late', 'T80_early', 'T80_late')
    for ax, pairs, title in zip(axes, (natural, rebuilt), ('A  |  Continuous flight', 'B  |  Rebuilt historical state')):
        for j, case in enumerate(cases):
            group = [p for p in pairs if p['case'] == case]
            contacts = 0
            for p in group:
                shift = (p['repeat']-1.5)*.05
                x = j-.13+shift
                y = 100*p['normal']['crossing_margin_m']
                ax.scatter(x, y, color=BLUE, marker='o', s=37, zorder=3,
                           label='Normal' if j == 0 and p['repeat'] == 1 else None)
                value = p['two_missed']['crossing_margin_m']
                if value is None:
                    contacts += (p['two_missed']['failure'] or {}).get('kind') == 'collision'
                else:
                    ax.plot([x, x+.26], [y, 100*value], color='#A5AEB8', linewidth=1.1)
                    ax.scatter(x+.26, 100*value, color=ORANGE, marker='s', s=37, zorder=3,
                               label='Two missed' if j == 0 and p['repeat'] == 1 else None)
            if contacts:
                ax.text(j, .11, f'{contacts}/{len(group)} contacts\nno crossing', ha='center',
                        transform=ax.get_xaxis_transform(), color=ORANGE, fontsize=9)
        ax.axhline(0, color='#666666', linewidth=.8, linestyle=':')
        ax.set(title=title, xticks=range(4), xlim=(-.45, 3.45), ylim=(-1, 11),
               xticklabels=['50 ms\nearly', '50 ms\nlate', '80 ms\nearly', '80 ms\nlate'])
    axes[0].set_ylabel('Gate 4 crossing clearance (cm)')
    axes[0].legend(frameon=False, loc='upper right', fontsize=8)
    fig.suptitle('Same two missed updates, different phase  |  AirSim FP32, two paired repeats per cell')
    save(fig, output, '03_phase_of_missed_updates')


def gain_figure(output):
    rows = load('teacher_gain_timing_airsim_20261110_131546/summary.json')['rows']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.35), layout='constrained')
    for gain, color, marker in zip((3.5, 7, 10), COLORS, MARKERS):
        by_t = sorted((r for r in rows if r['gain'] == gain and r['R_target_ms'] == 9), key=lambda r: r['T_ms'])
        by_r = sorted((r for r in rows if r['gain'] == gain and r['T_ms'] == 120), key=lambda r: r['R_target_ms'])
        for ax, selected, key, measure, scale in ((axes[0], by_t, 'T_ms', 'gate4_margin_m', 100),
                                                 (axes[1], by_r, 'R_target_ms', 'gate4_margin_m', 100),
                                                 (axes[2], by_r, 'R_target_ms', 'rmse_m', 1)):
            y = [scale*r[measure] if r[measure] is not None else np.nan for r in selected]
            ax.plot([r[key] for r in selected], y, label=f'Gain {gain:g}', color=color, marker=marker,
                    linewidth=1.6, markerfacecolor='white')
    axes[0].set(title='A  |  Fixed delay of 9 ms', xlabel='Period T (ms)', ylabel='Gate 4 clearance (cm)',
                xticks=[30, 60, 120])
    axes[1].set(title='B  |  Fixed period of 120 ms', xlabel='Target delay R (ms)',
                xticks=[9, 18, 24, 54, 99])
    axes[2].set(title='C  |  Fixed period of 120 ms', xlabel='Target delay R (ms)',
                ylabel='7.2 s tracking RMSE (m)', xticks=[9, 18, 24, 54, 99], ylim=(0, 1.08))
    for ax in axes[:2]:
        ax.set_ylim(-10, 103)
        ax.axhline(0, color='#666666', linewidth=.8, linestyle=':')
    axes[0].legend(frameon=False, loc='center left', fontsize=8)
    axes[1].text(.97, .23, 'Contact, no gate 4 crossing\nG3.5 at 54 & 99 ms\nG10 at 99 ms',
                 ha='right', transform=axes[1].transAxes, fontsize=8, color='#333333')
    fig.suptitle('MPC teacher in stepped AirSim  |  One run per setting, gaps indicate truncated flights')
    save(fig, output, '04_teacher_gain_sensitivity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.titlesize': 13, 'svg.fonttype': 'path'}):
        for plot in (action_figure, model_figure, phase_figure, gain_figure):
            plot(args.output)


if __name__ == '__main__':
    main()
