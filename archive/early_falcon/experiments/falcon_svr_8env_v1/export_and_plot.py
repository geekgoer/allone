"""Export all outcomes; draw measured eight-env cubes without filling missing data."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np

import experiment as exp


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ['record_id', 'environment_id', 'c', 'period_ms', 'R_max_ms', 'J_target', 'outcome']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(root):
    config = exp.read_json(root / 'experiment_manifest.json')['config']
    catalog = {m['model_id']: m for m in exp.read_json(root / 'model_manifest.json')['models']}
    plan = exp.read_json(root / 'plan.json')
    rows, censored = [], []
    for job in plan:
        directory = root / 'trials' / job['id']
        result = directory / 'result.json'
        final_raw = None
        if result.exists():
            final = exp.read_json(result)
            final_raw = final['raw_log']
            rows.append(exp.outcome_row(final, catalog[job['model_id']]))
        for attempt in directory.glob('attempt_*.summary.json'):
            record = exp.read_json(attempt)
            if record['raw_log'] != final_raw:
                row = exp.outcome_row(record, catalog[job['model_id']])
                row.update(valid_for_success_svr=False, valid_for_failure_zero_svr=False,
                    J_paper_style_success=None, J_failure_zero=None,
                    censoring_reason='infrastructure/cleanup/mixed-transport attempt; not a valid completed experiment')
                censored.append(row)
    output = root / 'datasets'
    write_table(output / 'all_recorded_trials.csv', rows)
    write_table(output / 'censored_infrastructure_attempts.csv', censored)
    with (output / 'all_recorded_trials.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    coverage = []
    for env in config['environments']:
        selected = [r for r in rows if r['environment_id'] == env['id']]
        directory = output / env['id']
        write_table(directory / 'all_trials.csv', selected)
        for policy, flag, target in (
                ('success_only', 'valid_for_success_svr', 'J_paper_style_success'),
                ('failure_zero', 'valid_for_failure_zero_svr', 'J_failure_zero')):
            ready = [dict(r, J_target=r[target], target_policy=policy) for r in selected if r[flag]]
            write_table(directory / f'svr_{policy}.csv', ready)
        coverage.append(dict(environment_id=env['id'], planned=sum(j['environment_id'] == env['id'] for j in plan),
            recorded=len(selected), completed=sum(r['completed'] for r in selected),
            collisions=sum(r['outcome'] == 'collision' for r in selected),
            missed_gates=sum(r['outcome'] == 'missed_gate' for r in selected),
            mission_timeouts=sum(r['outcome'] == 'mission_timeout' for r in selected),
            input_range_exceeded=sum(r['outcome'] == 'input_range_exceeded' for r in selected),
            success_svr_rows=sum(r['valid_for_success_svr'] for r in selected),
            failure_zero_svr_rows=sum(r['valid_for_failure_zero_svr'] for r in selected),
            distinct_recorded_coordinates=len({(r['model_id'], r['period_ms'], r['target_R_ms']) for r in selected})))
    report = dict(planned=len(plan), recorded=len(rows), censored_attempts=len(censored), environments=coverage,
        primary_performance_svr_target='E01..E08/svr_success_only.csv: original J on complete eligible routes',
        secondary_failure_penalty_target='E01..E08/svr_failure_zero.csv: engineering extension, not the primary performance J',
        features_for_per_environment_svr=['c', 'period_ms', 'R_max_ms'],
        optional_features_for_pooled_geometry_model=['c', 'period_ms', 'R_max_ms', 'x_spacing_m', 'lateral_step_m', 'vertical_step_m'],
        R_definition='Measured host state acquisition completion to AirSim send-call return, not target_R_ms or exact physics application time',
        success_only='Original J on completed pure-FPGA routes; conditional on completion and does not estimate collision risk',
        failure_zero='Engineering extension: physical collision/missed gate/mission timeout gets J_target=0. Infrastructure or mixed/tardy control is censored, not zero.',
        excluded_from_regression='All exclusions remain in all_trials and/or censored_infrastructure_attempts; partial J is diagnostic only',
        execution_mode='whole_model_nonpreemptive; segmented scheduling not validated')
    exp.write_json(output / 'dataset_manifest.json', report)
    print(f"EXPORT recorded={len(rows)}/{len(plan)} censored_attempts={len(censored)}", flush=True)
    return rows


def preview_layouts(config, output):
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 10}):
        nrows = (len(config['environments']) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(13, nrows*3.25), layout='constrained', squeeze=False)
        for env, ax in zip(config['environments'], axes.flat):
            for scenario in (s for s in exp.scenarios(config) if s['environment_id'] == env['id']):
                points = np.asarray(scenario['gate_centers'])
                variant = scenario['variant']
                ax.plot(points[:, 0], points[:, 1], '-o' if variant == 0 else '--s',
                    color='#0072B2' if variant == 0 else '#D55E00', label=f'Variant {variant}')
                if variant == 0:
                    for i, (x, y, _) in enumerate(points):
                        ax.annotate(str(i+1), (x, y), xytext=(3, 5), textcoords='offset points')
            ax.set(xlim=(0, 53), ylim=(-8, 8), xlabel='Forward x (m)', ylabel='Lateral y (m)',
                title=f"{env['id']} {env['name']}: dx={env['x_spacing_m']:g}, dy={env['lateral_step_m']:g}, dz={env['vertical_step_m']:g} m")
            ax.grid(alpha=.2)
        axes.flat[0].legend(loc='upper left')
        fig.suptitle('Frozen gate centers, not flight results\nAll gates are 4 m x 4 m; center heights alternate upward from NED z=-3 m')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor='white')
        plt.close(fig)


def plot_cubes(root, rows, policy):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    config = exp.read_json(root / 'experiment_manifest.json')['config']
    plan = exp.read_json(root / 'plan.json')
    catalog = {m['model_id']: m for m in exp.read_json(root / 'model_manifest.json')['models']}
    c_by_id = {i: 1.0/m['test_normalized_mse'] for i, m in catalog.items()}
    flag = 'valid_for_failure_zero_svr' if policy == 'failure_zero' else 'valid_for_success_svr'
    target = 'J_failure_zero' if policy == 'failure_zero' else 'J_paper_style_success'
    values = [r[target] for r in rows if r[flag]]
    norm = Normalize(vmin=0, vmax=max(values, default=0.))
    cmap = plt.get_cmap('viridis')
    output = root / 'figures' / policy
    output.mkdir(parents=True, exist_ok=True)
    rmax = max(config['target_responses_ms']) + config['response_tolerance_ms']
    for r in rows:
        if r['R_max_ms'] is not None and np.isfinite(r['R_max_ms']):
            rmax = max(rmax, r['R_max_ms'])
    with plt.rc_context({'font.size': 10}):
        for env in config['environments']:
            selected = [r for r in rows if r['environment_id'] == env['id']]
            present = {(r['model_id'], r['period_ms'], r['target_R_ms']) for r in selected}
            pending = {(j['model_id'], j['period_ms'], j['target_R_ms']) for j in plan if j['environment_id'] == env['id']} - present
            fig = plt.figure(figsize=(9.2, 8.4))
            ax = fig.add_axes([.02, .14, .80, .70], projection='3d')
            for success, marker, label in ((True, 'o', 'Completed'), (False, 'X', 'Physical mission failure')):
                group = [r for r in selected if r[flag] and r['completed'] == success]
                if group:
                    ax.scatter([r['c'] for r in group], [r['period_ms'] for r in group],
                        [r['R_max_ms'] for r in group], c=[r[target] for r in group], cmap=cmap, norm=norm,
                        marker=marker, edgecolors='#222222' if not success else 'none',
                        s=30 if success else 45, depthshade=False, label=label)
            invalid = [r for r in selected if not r[flag]]
            physical_without_J = [r for r in invalid if r['outcome'] in ('collision', 'missed_gate', 'mission_timeout')]
            if physical_without_J:
                ax.scatter([r['c'] for r in physical_without_J], [r['period_ms'] for r in physical_without_J],
                    [r['R_max_ms'] if r['R_max_ms'] is not None else r['target_R_ms'] for r in physical_without_J],
                    color='#D55E00', marker='X', s=35, label='Mission failure (no eligible J)')
            invalid = [r for r in invalid if r not in physical_without_J]
            if invalid:
                ax.scatter([r['c'] for r in invalid], [r['period_ms'] for r in invalid],
                    [r['R_max_ms'] if r['R_max_ms'] is not None else r['target_R_ms'] for r in invalid],
                    color='#777777', marker='^', s=22, label='Not eligible for this target')
            if pending:
                ax.scatter([c_by_id[m] for m, t, r in pending], [t for m, t, r in pending],
                    [r for m, t, r in pending], color='#BBBBBB', marker='.', s=7,
                    alpha=.5, label='Unmeasured (target coordinates)')
            ax.set(xlabel='Capability c (1 / test MSE)', ylabel='Period T (ms)',
                zlabel='Measured response R (ms)', xlim=(min(c_by_id.values()), max(c_by_id.values())),
                ylim=(min(config['periods_ms']), max(config['periods_ms'])),
                zlim=(min(config['target_responses_ms'])-1, rmax))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=24, azim=-55)
            ax.legend(loc='upper left', fontsize=8)
            if norm.vmax > 0:
                bar_ax = fig.add_axes([.87,.25,.025,.45])
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax,
                    label=f'Measured J ({policy}, shared scale)')
            else:
                fig.text(.5,.04,'No positive eligible J observed; no quantitative colorbar.',ha='center',fontsize=9)
            fig.suptitle(f"{env['id']} {env['name']} | {len(selected)}/{sum(j['environment_id']==env['id'] for j in plan)} recorded\nMeasured points only; c = 1 / test normalized MSE", y=.97)
            fig.savefig(output / f"cube_{env['id']}.png", dpi=180, facecolor='white')
            plt.close(fig)
    exp.write_json(output / 'figure_manifest.json', dict(target_policy=policy, source=str(root / 'datasets/all_recorded_trials.jsonl'),
        transformation='No interpolation; actual c/T/measured R per recorded trial; X marks physical mission failures',
        shared_color_limits=[norm.vmin, norm.vmax], c_layers=6,
        pending_coordinates='Planned target R only, gray markers; not observations',
        limitation='Only six measured capability levels. More display/interpolation points would not add observations.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=exp.HERE / 'results/moderate_v2')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--policy', choices=['success_only', 'failure_zero'], default='success_only')
    parser.add_argument('--preview-layouts', action='store_true')
    parser.add_argument('--config', type=Path, default=exp.HERE / 'collection_config_moderate_v2.json')
    parser.add_argument('--preview-output', type=Path, default=exp.HERE / 'layout_preview_moderate_v2.png')
    args = parser.parse_args()
    if args.preview_layouts:
        preview_layouts(exp.load_experiment(args.config)[0], args.preview_output)
        print('Layout preview:', args.preview_output)
        return
    root = args.root.resolve()
    rows = export_dataset(root)
    if args.plot:
        plot_cubes(root, rows, args.policy)


if __name__ == '__main__':
    main()
