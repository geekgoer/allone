"""Paired deterministic replay; phase inferred only from logged timestamps."""
import json
import argparse
import numpy as np
from collections import Counter
from dataclasses import replace
from pathlib import Path
from .bottleneck_joint import Job, Loop, Mode, Option, Platform, Scenario
from .bottleneck_abc import abc_repair
from .decision_dataset import atomic_json


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--diagnostic-p99', action='store_true', help='offline counterfactual only; not a certified bound')
    mode.add_argument('--split-post', action='store_true', help='compare separate command and worker completion budgets')
    args = parser.parse_args()
    root = Path('controller_architecture_study/results')
    if args.split_post:
        out = root/'split_post_replay_20261109_v1'
    else:
        out = root/('remaining_chain_p99_diagnostic_20261108_v1' if args.diagnostic_p99 else 'remaining_chain_replay_20261108_v1')
    out.mkdir(exist_ok=False)
    summaries = []
    for run in ('r0', 'r1'):
        p = root/'two_loop_adaptive_20261108_v1'/run
        records = json.loads((p/'decisions.json').read_text(encoding='utf-8'))
        rows = json.loads((p/'requests.json').read_text(encoding='utf-8'))
        prep = [1000*(x['feature_ready_s']-x['sample_start_s']) for x in rows if 'feature_ready_s' in x and 'sample_start_s' in x]
        post = [1000*(x['finished_s']-x['command_start_s']) for x in rows if 'finished_s' in x and 'command_start_s' in x]
        diagnostic_prep, diagnostic_post = float(np.percentile(prep, 99)), float(np.percentile(post, 99))
        by_key = {(r['vehicle'], r['cycle']): r for r in rows}
        results = []
        for i, r in enumerate(records):
            if i % 20 != 0 and r['status'] != 'installed':
                continue
            loops = tuple(Loop(**{**x, 'options': tuple(Option(o['period'], Mode(o['mode']['name'], tuple(o['mode']['segments'])), o['qualified']) for o in x['options'])}) for x in r['loop_context'])
            carry = tuple(Job(**x) for x in r['carry_context'])
            scenario = Scenario(loops, Platform(**r['platform']), horizon=240., carry=carry, initial_policy=('edf', 1), legal_groups=(1,))
            baseline = abc_repair(scenario)
            updated = []
            for job in carry:
                old = by_key[(job.loop, int(job.name.rsplit(':', 1)[1]))]
                t = r['snapshot_s']
                operations = None
                if old.get('board_done_s', float('inf')) <= t or old.get('board_start_s', float('inf')) <= t:
                    operations = (('ps', scenario.platform.post),)
                elif old.get('queue_s', float('inf')) <= t:
                    operations = (('pl', sum(job.segments)), ('ps', scenario.platform.post))
                updated.append(replace(job, remaining_operations=operations))
            fixed = abc_repair(replace(scenario, carry=tuple(updated)))
            if args.split_post:
                t = r['snapshot_s']
                acknowledgements = [x for x in rows if x.get('command_return_s', float('inf')) <= t]
                post_ms = max([2.] + [1000*(x['command_return_s']-x['command_start_s']) for x in acknowledgements])
                rearm_ms = max([2.] + [1000*(x['finished_s']-x['command_return_s']) for x in acknowledgements
                                       if x.get('finished_s', float('inf')) <= t])
                platform = replace(scenario.platform, post=post_ms, rearm=rearm_ms)
                split_carry = []
                for job in updated:
                    old = by_key[(job.loop, int(job.name.rsplit(':', 1)[1]))]
                    if old.get('command_return_s', float('inf')) <= t:
                        split_carry.append(replace(job, remaining_operations=(), command_completed=True))
                    elif job.remaining_operations is not None:
                        operations = tuple((k, post_ms if k == 'ps' else v) for k, v in job.remaining_operations)
                        split_carry.append(replace(job, remaining_operations=operations))
                    else:
                        split_carry.append(job)
                split = abc_repair(replace(scenario, platform=platform, carry=tuple(split_carry)))
            if args.diagnostic_p99:
                platform = replace(scenario.platform, prep=diagnostic_prep, post=diagnostic_post)
                recosted = tuple(replace(j, remaining_operations=tuple((k, diagnostic_post if k == 'ps' else v) for k, v in j.remaining_operations)) if j.remaining_operations is not None else j for j in updated)
                diagnostic = abc_repair(replace(scenario, platform=platform, carry=recosted))
            results.append(dict(index=i, status=r['status'], old_accepted=baseline['accepted'],
                                revised_accepted=fixed['accepted'], changed_carry=sum(a != b for a, b in zip(carry, updated)),
                                old_stop=baseline['stop_reason'], revised_stop=fixed['stop_reason'],
                                split_passed=pipeline_passed(split) if args.split_post else None,
                                original_pipeline_passed=pipeline_passed(baseline) if args.split_post else None,
                                split_response_passed=split['accepted'] if args.split_post else None,
                                diagnostic_accepted=diagnostic['accepted'] if args.diagnostic_p99 else None))
        atomic_json(out/f'{run}.json', results)
        summary = dict(run=run, sampled=len(results), old_accepted=sum(x['old_accepted'] for x in results),
                       revised_accepted=sum(x['revised_accepted'] for x in results),
                       recovered=sum(not x['old_accepted'] and x['revised_accepted'] for x in results),
                       revised_stops=dict(Counter(x['revised_stop'] for x in results)))
        if args.split_post:
            summary.update(original_pipeline_passed=sum(x['original_pipeline_passed'] for x in results),
                           split_response_passed=sum(x['split_response_passed'] for x in results),
                           split_pipeline_passed=sum(x['split_passed'] for x in results))
        if args.diagnostic_p99:
            summary.update(diagnostic_p99_prep_ms=diagnostic_prep, diagnostic_p99_post_ms=diagnostic_post,
                           diagnostic_accepted=sum(x['diagnostic_accepted'] for x in results),
                           scope='offline hindsight p99, not a hard realtime bound')
        summaries.append(summary)
        print(summary, flush=True)
    atomic_json(out/'summary.json', summaries)


def pipeline_passed(result):
    if not result['accepted']:
        return False
    forecast = result['forecast']
    previous = {}
    for job in sorted(forecast['jobs'], key=lambda j: (j.loop, j.release)):
        if previous.get(job.loop, float('-inf')) > job.release+1e-8:
            return False
        previous[job.loop] = forecast['finish'][job.name]
    return True


if __name__ == '__main__':
    main()
