"""AirSim board-in-the-loop trial of model selection under a supplied service budget."""
import argparse
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json
from .dynamic_probe_common import read_json
from .fast_convex_preprocess import encode
from .framework_runtime_world import FrameworkWorld
from .ip37_shadow import make_client
from .probe_main18_historical_models import DELAY, PERIOD, cases, compare
from .probe_main18_matched_recovery import airsim_trial
from .profile_main18_board import PACK, select
from .safety_comparison_core import BODY_RADIUS, CLEARANCE


def run(profile_path, results, pack, output):
    profile = read_json(profile_path)
    budgets = (.10, .20)
    fast, precise = (select(profile, value) for value in budgets)
    if fast is None or precise is None or fast['id'] == precise['id']:
        raise ValueError('the service budgets must expose two distinct eligible models')
    plan = [case for case in cases(results, True) if case[0] in ('T50_early', 'independent_reverse_turn')]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'setup.json', dict(profile=str(profile_path), budgets_ms=budgets,
        fast=fast, precise=precise, budget_switch_s=.4, step_period_s=PERIOD, target_delay_s=DELAY,
        workload='supplied service budget; stepped AirSim with live INT8 inference'))
    client, manifest = make_client(pack, '192.168.77.2', 5237, 1500)
    pairs = []
    try:
        if manifest['wire_tag_hex'] != profile['wire_tag']:
            raise ValueError('AirSim probe board differs from profiled board bundle')
        for label, info, layout, scenario, gates, duration in plan:
            world = FrameworkWorld(layout, info['trace_phase_s'], 1, info['width_m'], task_duration=duration)
            try:
                if not set(gates).issubset(world.expected['Drone1']):
                    raise ValueError(f'{label}: target gates are missing')
                for repeat, order in enumerate((('fast', 'adaptive', 'precise'), ('precise', 'adaptive', 'fast')), 1):
                    arms = {}
                    for policy in order:
                        requests = []

                        def control(state, refs):
                            budget = budgets[len(requests) >= round(.4/PERIOD)]
                            model = select(profile, budget) if policy == 'adaptive' else (fast if policy == 'fast' else precise)
                            reply = client.infer(model['id'], encode(state, refs).tolist())
                            requests.append(dict(budget_ms=budget, **reply))
                            if reply['status'] or reply['saturated_input_channels']:
                                raise RuntimeError(f'board rejected input for {label} / {policy}')
                            return np.clip(reply['U_raw_physical'], [2., -3., -3., -2.], [20., 3., 3., 2.])

                        directory = output/label/f'rep{repeat}'/policy
                        try:
                            trial, commands, trace = airsim_trial(world, None, policy, scenario, directory, duration,
                                release_delays={0: DELAY}, period=PERIOD, state_controller=control)
                        finally:
                            atomic_json(directory/'board_requests.json', requests)
                        physical = trial['physical']
                        margins = {}
                        for crossing in physical['crossings']:
                            if crossing['gate'] in gates:
                                margins[crossing['gate']] = min(info['width_m']/2-abs(crossing['lateral']),
                                    info['height_m']/2-abs(crossing['vertical']))-BODY_RADIUS-CLEARANCE
                        measured_misses = sum(r['board_service_us'] > r['budget_ms']*1000 for r in requests)
                        row = dict(policy=policy, gate_margins_m=margins, failure=physical['failure'],
                            target_pass=len(margins)==len(gates) and min(margins.values())>=0
                            and physical['failure'] is None and physical['window_covered'],
                            service_budget_misses=measured_misses, count=len(requests),
                            model_ids=sorted({r['requested_model_id'] for r in requests}),
                            service_total_ms=sum(r['board_service_us'] for r in requests)/1000,
                            tracking_rmse_m=trial['position_rmse_m'])
                        atomic_json(directory/'result.json', row)
                        atomic_json(directory/'commands.json', commands)
                        atomic_json(directory/'trace.json', trace)
                        arms[policy] = (trial, commands, row)
                        print(label, repeat, row, flush=True)
                    trial, commands, _ = arms['fast']
                    pairs.append(dict(case=label, repeat=repeat, arms=[dict(row, **compare(trial, a, commands, b))
                                      for a,b,row in arms.values()]))
                    atomic_json(output/'summary.json', dict(complete=len(pairs)==2*len(plan), pairs=pairs))
            finally:
                atomic_json(output/label/'cleanup.json', world.close())
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=PACK)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.results, args.pack, args.output)
