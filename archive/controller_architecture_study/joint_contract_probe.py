"""Offline decision/cost probe, using frozen flight priors and measured costs.

Deadline changes below are declared synthetic inputs, not a collision rescue.
No AirSim connection, policy installation, controller training or test-set read.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import choose_layout, cpu_topology, set_process_cpus
from .joint_contract_optimizer import compile_options, optimize


ROOT = Path(__file__).resolve().parent


def load_priors(path):
    """Use measured short-J only; virtual-prefix J_hat is not consumed."""
    values = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in read_json(path)['rows']:
        result = row.get('result') or {}
        if not result.get('complete'):
            continue
        config = row['configuration']
        for v, metrics in result['per_vehicle'].items():
            value = float(metrics['short_J'])
            if np.isfinite(value):
                values[v][config['models'][v]][config['periods_ms'][v]/1000].append(value)
    return {v:{m:sorted((t,float(np.mean(samples))) for t,samples in periods.items())
               for m,periods in models.items()} for v,models in values.items()}


def make_case(priors, profile, costs, deadlines):
    future, options = {}, {}
    service = sum(costs.get(k,0.) for k in
        ('sample_io_s','input_s','ingress_s','egress_s','handoff_s','command_s','completion_guard_s'))
    for i,v in enumerate(sorted(priors,key=lambda v:int(v.removeprefix('Drone')))):
        previous = -.002*i
        deadline = previous+deadlines.get(v,.110)
        future[v] = dict(vehicle=v, sample_id=v+':next', model='plain_mlp_h9_w1024_seed0', priority=i,
            previous_release_s=previous, minimum_release_s=previous+.020,
            planned_release_s=previous+.030, requested_period_s=.030, deadline_s=deadline)
        actions = []
        for model, knots in priors[v].items():
            periods, values = zip(*knots)
            # Measured knots plus one deadline-derived point. Interpolation is
            # inside observed brackets only and is explicitly a nominal prior.
            latest = deadline-service-sum(profile[model]['segment_estimates_s'])-previous
            candidates = sorted(set([*periods, latest] if periods[0] <= latest <= periods[-1] else periods))
            for period in candidates:
                measured = any(abs(period-t)<1e-10 for t in periods)
                actions.append(dict(id=f'{model}/T{period*1000:.6f}', model=model,
                    release_s=previous+period, value=float(np.interp(period,periods,values)),
                    source='measured_short_J_mean' if measured else 'bracketed_linear_short_J_prior',
                    quality_scope='nominal-route prior; neither recovery qualification nor state-conditioned future J'))
        options[v] = actions
    return future, options


def run(output, repeats):
    output.mkdir(parents=True,exist_ok=False)
    affinity = set_process_cpus(choose_layout(cpu_topology(),psutil.Process().cpu_affinity())['frontend_cpus'])
    source = ROOT/'results/fair_pool_search_20261028_v1/strengthened/search_progress.json'
    timing = ROOT/'results/joint_runtime_repair_20261029_v3/pipeline_costs.json'
    profile_file = ROOT/'results/joint_runtime_repair_20261029_v3/seed4_joint/report.json'
    priors, costs = load_priors(source), read_json(timing)['costs']
    profile = read_json(profile_file)['worker_profile']
    cases = dict(normal={}, two_tight_deadlines={'Drone1':.035,'Drone3':.035},
                 all_deadlines_too_tight={v:.022 for v in priors})
    results = {}
    with threadpool_limits(limits=1):
        for name, deadlines in cases.items():
            future, options = make_case(priors,profile,costs,deadlines)
            kwargs = dict(inflight_limit=10,planning_budget_s=.006)
            # Warm code/data before measuring. This does not modify a policy.
            optimize(0.,future,[],options,profile,costs,**kwargs)
            planning, accepted, samples = [], 0, []
            proposal, report = None, None
            for _ in range(repeats):
                start = time.perf_counter()
                p,r = optimize(0.,future,[],options,profile,costs,**kwargs)
                planning.append(1000*(time.perf_counter()-start))
                accepted += r['accepted']
                if len(samples)<3 and not r['accepted']:
                    samples.append(r)
                if report is None or r['accepted']:
                    proposal, report = p,r
            vehicles, choices, rejected = compile_options(0.,future,options,profile,costs,.006,.020)
            local = None
            if all(choices):
                ix = [int(np.argmax([a['value'] for a in row])) for row in choices]
                local = dict(value=sum(row[j]['value'] for row,j in zip(choices,ix)),
                    load=np.sum([row[j]['demand'] for row,j in zip(choices,ix)],axis=0).tolist(),
                    decisions={v:dict(model=row[j]['model'],period_ms=1000*row[j]['period_s'])
                               for v,row,j in zip(vehicles,choices,ix)},
                    scope='independent utility maximization, not globally admitted')
            elapsed = np.array(planning)
            results[name] = dict(proposal=proposal,report=report,unconstrained_local=local,
                input_future=future,input_options=options,accepted_runs=accepted,repeats=repeats,
                rejected_examples=samples,planning_ms=dict(mean=float(elapsed.mean()),
                    p99=float(np.quantile(elapsed,.99)),max=float(elapsed.max())),
                action_count=sum(map(len,options.values())),synthetic_deadline_overrides=deadlines)
            print(f'{name}: accepted={accepted}/{repeats}; mean={elapsed.mean():.3f}ms '
                  f'p99={np.quantile(elapsed,.99):.3f}ms max={elapsed.max():.3f}ms',flush=True)
            if report['accepted']:
                for v,d in report['decisions'].items():
                    print(f"  {v}: {d['model']} T={d['interval_ms']:.3f}ms",flush=True)
            else:
                print('  '+report['reason'],flush=True)
    result = dict(scope='offline decision mechanism and empirical CPU timing; not flight or physical safety validation',
        source_files={str(p):sha256(p) for p in (source,timing,profile_file)},affinity=affinity,
        test_read=False,neural_policy_used=False,actual_future_J_measured=False,
        prior_limitations='nominal per-loop short-J measured under different joint loads; separability and interpolation are approximations',
        live_policy_changed=False,planning_reserve_ms=6.,cases=results)
    atomic_json(output/'report.json',result)
    print('Report: '+str(output/'report.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=200)
    args = parser.parse_args()
    if args.repeats<1:
        parser.error('repeats must be positive')
    run(args.output,args.repeats)


if __name__ == '__main__':
    main()
