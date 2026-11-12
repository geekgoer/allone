"""Real-board finite static search, then frozen static/adaptive paired flights."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import describe, read_json
from .ip37_safety_cases import make_case, audit


PERIODS = (20., 30., 40., 50., 60., 80., 100., 120.)
ROOT = Path(__file__).resolve().parents[1]


def profile_key(profile):
    return json.dumps(profile, sort_keys=True)


def neighbors(profile, roles):
    """Coordinate-greedy neighborhood: one role changes one period step, or order changes."""
    for role in ('early', 'middle', 'late', 'smooth'):
        names = [v for v in profile['periods'] if roles[v]['role'] == role]
        if not names:
            continue
        index = PERIODS.index(profile['periods'][names[0]])
        for j in (index-1, index+1):
            if not 0 <= j < len(PERIODS):
                continue
            candidate = dict(periods=dict(profile['periods']), order=profile['order'])
            for name in names:
                candidate['periods'][name] = PERIODS[j]
            yield candidate
    yield dict(periods=dict(profile['periods']), order='fifo' if profile['order'] == 'edf' else 'edf')


def best_static(rows):
    valid = [r for r in rows if r['comparable'] and r['J_total'] is not None]
    timely = [r for r in valid if r['timing']]
    # Preserve the best measured J even if Windows provided no timely candidate;
    # its timing qualification remains false and is reported explicitly.
    pool = timely or valid
    return max(pool, key=lambda r: (r['J_total'], -r['requests'])) if pool else None


def trial_summary(directory, profile, policy):
    r, setup = read_json(directory/'result.json'), read_json(directory/'setup.json')
    requests, decisions = read_json(directory/'requests.json'), read_json(directory/'decisions.json')
    evaluation = r.get('evaluation', {})
    contacts = {v: x['contact'] for v, x in r['vehicles'].items() if x.get('contact')}
    passed = [v for v, x in r['vehicles'].items() if x['error'] is None and x['window_covered']
              and x['body_clearance_passed'] and all(g['valid'] for g in x['crossings'])]
    origin = setup['origin_s']
    contact_times = [c['observation_t_s'] for c in contacts.values() if 'observation_t_s' in c]
    applied = [d for d in decisions if d['status'] == 'installed']
    orders = [profile['order']]+[d['policy'][0] for d in applied]
    gaps, actual_periods, service = {}, {}, []
    for name in setup['vehicles']:
        jobs = [j for j in requests if j['vehicle'] == name]
        acks = sorted(j['command_return_s'] for j in jobs if 'command_return_s' in j)
        gaps[name] = describe([1000*(b-a) for a, b in zip(acks, acks[1:])])
        actual_periods[name] = sorted(set(j['period_ms'] for j in jobs))
    for job in requests:
        if (job.get('board_timing') or {}).get('board_service_us') is not None:
            service.append(job['board_timing']['board_service_us']/1000)
    full_j = evaluation.get('total_J')
    global_errors = [e for e in r['errors'] if not e.startswith(tuple(v+':' for v in setup['vehicles']))]
    comparable = (bool(evaluation.get('valid')) and not global_errors and r['sim_clock_passed']
                  and r['calendar_coverage_passed'])
    return dict(directory=str(directory), profile=profile, policy=policy, comparable=comparable,
        J_total=full_j, J_per_loop=full_j/len(setup['vehicles']) if full_j is not None else None,
        flight=r['flight_passed'], body=r['body_clearance_passed'],
        timing=r['timing_before_contact_passed'], clock=r['sim_clock_passed'],
        collisions=len(contacts), contacts=contacts, passed_vehicles=passed,
        completed_loops=len(passed), requests=r['requests'], outcomes=r['outcomes'],
        installed=len(applied), period_updates=r['period_updates'],
        order_switches=sum(a != b for a, b in zip(orders, orders[1:])),
        actual_periods_ms=actual_periods, response_ms=r['response_ms'], command_gaps_ms=gaps,
        board_queue_ms=r['board_queue_ms'], board_service_ms=describe(service),
        decision_ms=r['decision_ms'], decisions=r['decisions'], snapshot_skips=r['snapshot_skips'],
        planner_service_ms=r.get('planner_service_ms'), planner_thread_cpu_ms=r.get('planner_thread_cpu_ms'),
        pre_contact_outcomes=dict(Counter(j['status'] for j in requests if not contact_times or
                                         j['release_s']-origin < min(contact_times))),
        post_contact_outcomes=dict(Counter(j['status'] for j in requests if contact_times and
                                          j['release_s']-origin >= min(contact_times))),
        conditional_physical_misses=r['conditional_physical_deadline_misses'],
        first_contact_t_s=min(contact_times) if contact_times else None,
        first_install_t_s=applied[0]['done_s']-origin if applied else None,
        source_sha256=setup['source_sha256'], case_sha256=setup['case_sha256'],
        model_checkpoint_sha256=setup['model'].get('source_checkpoint_sha256'),
        wire_tag_hex=setup['wire_tag_hex'], evaluation=evaluation, errors=r['errors'],
        final_board_envelope_ms=r['final_empirical_board_cost_ms'],
        final_cpu_envelope_ms=r['final_empirical_cpu_cost_ms'],
        workload_rule='contact retires only that loop; observer continues; postfailure load is reduced')


def paired(fixed, adaptive):
    if (fixed['case_sha256'], fixed['wire_tag_hex'], fixed['profile']) != (
            adaptive['case_sha256'], adaptive['wire_tag_hex'], adaptive['profile']):
        raise ValueError('paired case, board or initial configuration differs')
    return dict(fixed=fixed, adaptive=adaptive,
        comparable=fixed['comparable'] and adaptive['comparable'],
        delta_J=adaptive['J_total']-fixed['J_total'] if fixed['comparable'] and adaptive['comparable'] else None,
        delta_collisions=adaptive['collisions']-fixed['collisions'],
        rescued_vehicles=sorted(set(fixed['contacts']) & set(adaptive['passed_vehicles'])),
        newly_failed_vehicles=sorted(set(fixed['passed_vehicles'])-set(adaptive['passed_vehicles'])),
        installed_before_static_contact=(adaptive['first_install_t_s'] < fixed['first_contact_t_s']
            if adaptive['first_install_t_s'] is not None and fixed['first_contact_t_s'] is not None else None))


def save_tables(root, summary):
    atomic_json(root/'summary.json', summary)
    trials, pairs, cells = [], [], []
    for cell in summary['cells']:
        tag = {k: cell[k] for k in ('drones', 'net_width_m', 'seed')}
        for stage in ('search', 'evaluation'):
            for row in cell.get(stage, []):
                trials.append(dict(**tag, stage=stage, policy=row['policy'], directory=row['directory'],
                    profile=json.dumps(row['profile'], sort_keys=True), comparable=row['comparable'],
                    J=row['J_total'], collisions=row['collisions'], completed=row['completed_loops'],
                    timing=row['timing'], requests=row['requests'], installed=row['installed'],
                    period_updates=row['period_updates'], R_p99_ms=(row['response_ms'] or {}).get('p99'),
                    R_max_ms=(row['response_ms'] or {}).get('max_observed'),
                    max_command_gap_ms=max(((v or {}).get('max_observed') or 0 for v in row['command_gaps_ms'].values()), default=0)))
        for i, p in enumerate(cell.get('pairs', [])):
            pairs.append(dict(**tag, repeat=i, comparable=p['comparable'],
                fixed_J=p['fixed']['J_total'], adaptive_J=p['adaptive']['J_total'], delta_J=p['delta_J'],
                fixed_collisions=p['fixed']['collisions'], adaptive_collisions=p['adaptive']['collisions'],
                rescued=','.join(p['rescued_vehicles']), newly_failed=','.join(p['newly_failed_vehicles'])))
        for policy in ('fixed', 'adaptive'):
            rows = [r for r in cell.get('evaluation', []) if r['policy'] == policy]
            scores = [r['J_total'] for r in rows if r['comparable']]
            cells.append(dict(**tag, policy=policy, trials=len(rows), comparable=len(scores),
                mean_J=sum(scores)/len(scores) if scores else None,
                collisions=sum(r['collisions'] for r in rows), completed_loops=sum(r['completed_loops'] for r in rows),
                timing_passed=sum(r['timing'] for r in rows), installed=sum(r['installed'] for r in rows)))
    for name, rows in (('trials.csv', trials), ('pairs.csv', pairs), ('cells.csv', cells)):
        if rows:
            with (root/name).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def run_trial(args, case_path, profile, policy, folder, expected_sources):
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(case_sha256=sha256(case_path), profile=profile, policy=policy,
                    source_sha256=expected_sources, bundle_sha256=sha256(args.bundle/'manifest.json'),
                    evaluation_ms=args.evaluation_ms, model_id=11)
    saved = folder/'trial_summary.json'
    if saved.exists():
        if read_json(folder/'trial.json') != metadata:
            raise ValueError(f'cannot reuse mismatched trial: {folder}')
        return read_json(saved)
    atomic_json(folder/'trial.json', metadata)
    atomic_json(folder/'initial_periods.json', profile['periods'])
    attempt = folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:03d}'
    command = [sys.executable, '-u', '-m', 'controller_architecture_study.fly_ip37_joint',
        '--bundle', str(args.bundle), '--board', args.board, '--board-port', str(args.port),
        '--airsim-host', args.airsim_host, '--case', str(case_path), '--model-id', '11',
        '--policy', policy, '--order', profile['order'], '--period-ms', '80',
        '--initial-periods', str(folder/'initial_periods.json'), '--evaluation-ms', str(args.evaluation_ms),
        '--periods-ms', *map(str, PERIODS), '--vehicles', *profile['periods'], '--output', str(attempt)]
    atomic_json(folder/'command.json', command)
    print(f'RUN {folder.name}: {policy} {profile}', flush=True)
    try:
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace', timeout=180)
    except subprocess.TimeoutExpired as exc:
        (folder/'console_timeout.txt').write_bytes(exc.stdout or b'')
        raise RuntimeError(f'trial timed out; inspect {folder}; restart Play before resuming') from exc
    (folder/f'{attempt.name}_console.txt').write_text(completed.stdout, encoding='utf-8')
    print(completed.stdout, flush=True)
    if not (attempt/'setup.json').exists() or not (attempt/'result.json').exists():
        raise RuntimeError(f'incomplete setup/result, inspect {folder}; preserve reports and resume after repair')
    row = trial_summary(attempt, profile, policy)
    if row['source_sha256'] != {k: expected_sources[k] for k in row['source_sha256']}:
        raise ValueError('flight source changed during campaign')
    # Collisions and real deadline misses remain results. Infrastructure failures
    # stop the batch instead of repeating hundreds of invalid flights.
    fatal = ('golden', 'hash mismatch', 'Play must', 'gate cleanup', 'gates remain',
             'Connection', 'Timeout', 'timeout', 'unpause', 'launch reserve', 'drain:')
    if not row['clock'] or any(any(key in e for key in fatal) for e in row['errors']):
        raise RuntimeError(f'infrastructure/clock failure; inspect {attempt}; keep Unreal visible and resume after repair')
    atomic_json(saved, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--board', default='192.168.77.2')
    parser.add_argument('--port', type=int, default=5238)
    parser.add_argument('--airsim-host', default='127.0.0.1')
    parser.add_argument('--drones', type=int, nargs='+', default=[4, 7, 10, 12])
    parser.add_argument('--widths', type=float, nargs='+', default=[1.9, 2.4, 3.0])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--search-budget', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--evaluation-ms', type=float, default=50.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if (any(n < 4 or n > 12 for n in args.drones) or len(set(args.drones)) != len(args.drones)
            or any(not 1.7 <= w <= 3. for w in args.widths) or len(set(args.widths)) != len(args.widths)
            or any(s not in (0, 1, 2) for s in args.seeds) or len(set(args.seeds)) != len(args.seeds)
            or args.search_budget < 3 or args.repeats < 1 or not 20 <= args.evaluation_ms <= 50):
        parser.error('unique N=4..12, widths=1.7..3m, seeds=0..2, search budget>=3, repeats>=1, evaluation=20..50ms required')
    sources = {name: sha256(Path(__file__).with_name(name)) for name in (
        'ip37_campaign.py', 'ip37_evaluation.py', 'ip37_safety_cases.py', 'fly_ip37_joint.py',
        'ip37_joint_runtime.py', 'bottleneck_abc.py', 'bottleneck_joint.py',
        'held_control_safety.py', 'convex_teacher.py', 'safety_comparison_core.py', 'fast_convex_preprocess.py')}
    settings = dict(drones=args.drones, widths=args.widths, seeds=args.seeds,
        search_budget=args.search_budget, repeats=args.repeats, evaluation_ms=args.evaluation_ms,
        bundle=str(args.bundle.resolve()), bundle_sha256=sha256(args.bundle/'manifest.json'),
        board=args.board, port=args.port, airsim_host=args.airsim_host, sources=sources,
        metric='sum of Eq4-form J; Wx=I10; common20ms interpolation; observed post-contact states included',
        static_selection='largest calibration J among observed timely candidates; if none, explicitly unqualified best J',
        baseline='finite role-coordinate greedy, not original FALCON SVM or global optimum',
        model_switching=False, preemption=False, original_teacher_unchanged=True)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.output/'campaign.json'
    if manifest_path.exists() and read_json(manifest_path) != settings:
        raise ValueError('campaign settings/source/bundle differ; use a new output directory')
    atomic_json(manifest_path, settings)
    cases = {}
    for seed in args.seeds:
        for width in args.widths:
            path = args.output/'cases'/f's{seed}_w{width:g}.json'
            if not path.exists():
                case = make_case(seed, width, vehicles=12)
                check = audit(case)
                atomic_json(path.with_name(path.stem+'_audit.json'), check)
                if not check['passed']:
                    raise ValueError(f'reference geometry audit failed: {path}')
                atomic_json(path, case)
            cases[seed, width] = path
    cells = len(args.drones)*len(args.widths)*len(args.seeds)
    print(f'PLAN {cells} cells; at most {cells*(args.search_budget+2*args.repeats)} flights; '
          f'flight-only hours <= {cells*(args.search_budget+2*args.repeats)*26/3600:.2f}; setup/drain extra', flush=True)
    atomic_json(args.output/'plan.json', dict(cells=cells,
        maximum_flights=cells*(args.search_budget+2*args.repeats), duration_s=26.,
        cases={f's{s}_w{w:g}': dict(file=str(p), sha256=sha256(p)) for (s, w), p in cases.items()},
        plan_only=args.plan_only))
    if args.plan_only:
        return
    summary = dict(cells=[], scope=settings, status='running')
    try:
        for seed in args.seeds:
            for width in args.widths:
                case_path = cases[seed, width]
                roles = read_json(case_path)['roles']
                for n in args.drones:
                    names = [f'Drone{i+1}' for i in range(n)]
                    folder = args.output/f's{seed}_w{width:g}_n{n:02d}'
                    cell = dict(seed=seed, net_width_m=width, drones=n, search=[], evaluation=[], pairs=[])
                    summary['cells'].append(cell)
                    seen = set()

                    def evaluate(profile):
                        seen.add(profile_key(profile))
                        row = run_trial(args, case_path, profile, 'fixed',
                            folder/f'search_{len(cell["search"]):02d}', sources)
                        cell['search'].append(row)
                        save_tables(args.output, summary)

                    for t in (30., 60., 120.):
                        evaluate(dict(periods=dict.fromkeys(names, t), order='edf'))
                    while len(cell['search']) < args.search_budget:
                        best = best_static(cell['search'])
                        if best is None:
                            break
                        proposals = [p for p in neighbors(best['profile'], roles) if profile_key(p) not in seen]
                        if not proposals:
                            break
                        for proposal in proposals[:args.search_budget-len(cell['search'])]:
                            evaluate(proposal)
                    selected = best_static(cell['search'])
                    if selected is None:
                        cell['status'] = 'no_comparable_static'
                        save_tables(args.output, summary)
                        continue
                    cell['selected'] = selected
                    cell['static_timing_qualified'] = selected['timing']
                    atomic_json(folder/'selected_static.json', selected)
                    for repeat in range(args.repeats):
                        results = {}
                        for policy in (('fixed', 'adaptive') if repeat % 2 == 0 else ('adaptive', 'fixed')):
                            row = run_trial(args, case_path, selected['profile'], policy,
                                folder/f'r{repeat:02d}_{policy}', sources)
                            results[policy] = row
                            cell['evaluation'].append(row)
                            save_tables(args.output, summary)
                        cell['pairs'].append(paired(results['fixed'], results['adaptive']))
                        save_tables(args.output, summary)
                    cell['status'] = 'completed'
                    save_tables(args.output, summary)
        summary['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_tables(args.output, summary)
        print(f'Campaign summary: {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
