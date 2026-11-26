"""Run the native Zynq PS controller against the isolated onboard Blocks."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from config import DEFAULT_CALIBRATION_PATH
from .run_onboard_fleet_value import trial_margins
from .run_onboard_load_replay import load_windows
from .run_onboard_pilot import scenario, trial
from .run_onboard_stagger_validation import LEADING, ROUTES
from .run_onboard_value import PORT, read, save


STAGGER_S = .90
WINDOW_STARTS = (1, 9, 17)
REQUEST_BUDGET = 18
STATIC_CANDIDATES = ((2, 5), (2, 11), (3, 5), (3, 11))
BOARD_PACKAGE = '/home/root/board_package_requant18_release001'


def ssh_command(args):
    remote = shlex.join([args.binary, args.windows_ip, args.package])
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
    board = [*command, args.board, 'exec ' + remote]
    return [*command, args.ssh_hop, shlex.join(board)] if args.ssh_hop else board


def fly(args, client, gates, path, amplitudes, distances, slots, probes, policy, period, model):
    spec = dict(policy=policy, slots=slots, request_budget=REQUEST_BUDGET, fixed_period_slots=period,
                fixed_model=model, port=PORT, calibration=read(DEFAULT_CALIBRATION_PATH), probes=probes,
                fleet=[dict(vehicle=v, amplitude=a) for v, a in amplitudes.items()])
    log = path.with_suffix('.ps.log')
    with log.open('w', encoding='utf-8') as diagnostic:
        with subprocess.Popen(ssh_command(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=diagnostic, text=True, encoding='utf-8') as ps:
            try:
                ps.stdin.reconfigure(newline='\n')
                ps.stdin.write(json.dumps(spec)+'\n')
                ps.stdin.flush()
                while True:
                    line = ps.stdout.readline()
                    if not line:
                        raise RuntimeError(f'PS did not reach READY. See {log}')
                    if line.startswith('{'):
                        ready = json.loads(line)
                        if ready.get('ready'):
                            break
                    diagnostic.write(line)

                def feed(stop):
                    stdout, _ = ps.communicate(input='GO\n', timeout=12)
                    if ps.returncode:
                        raise RuntimeError(f'PS trial failed. See {log}')
                    native = json.loads(stdout.splitlines()[-1])
                    save(path/'native_ps.json', native)
                    return {}, native['requests']

                result = trial(client, PORT, path, gates, 3, False, feed, leading=LEADING, stagger_s=STAGGER_S,
                               amplitudes=amplitudes)
            finally:
                if ps.poll() is None:
                    ps.kill()
    native = read(path/'native_ps.json')
    summary = inspect(path, result, native, probes)
    summary.update(policy=policy, fixed_period_ms=period*180, fixed_model=model, golden_max_abs=ready['goldens'],
                   **trial_margins(path, amplitudes=amplitudes, distances=distances))
    save(path/'joined_report.json', summary)
    print(f"{path.name} requests={summary['requests']} adopted={summary['adopted']} "
          f"key={summary['key_adopted']} margin={summary['fleet_min_m']} pass={summary['pass']}", flush=True)
    return summary


def inspect(path, result, native, probes):
    traces = {v: read(path/f'{v}.json') for v in result}
    requests = {(r['vehicle'], r['slot'], r['sample_ns']): r for r in native['requests']}
    applied = []
    for v, trace in traces.items():
        for actual, x, _, _, source, seq, sample in trace:
            if source != 3:
                continue
            row = requests.get((v, seq, sample))
            if not row or not row['accepted'] or not 0 <= actual-sample <= 55_000_000:
                raise RuntimeError('Found an unpaired or expired adopted command')
            timely = any(abs(x-probes[p]) <= .25 for p in ('g2_early', 'g3_early', 'g3_near'))
            applied.append(dict(vehicle=v, slot=int(seq), age_ms=(actual-sample)/1e6, x_m=x, timely=timely))
    in_between = {}
    for v, trace in traces.items():
        used = [r for r in trace if r[4] == 3]
        if len(used) >= 2:
            in_between[v] = sum(r[4] == 1 and used[0][0] < r[0] < used[1][0] for r in trace)
    return dict(vehicles=result, requests=len(requests), accepted=sum(r['accepted'] for r in requests.values()),
                adopted=len(applied), key_adopted=sum(r['timely'] for r in applied),
                dpu_run_ms=sum(r.get('dpu_run_us',0) for r in requests.values())/1000,
                ps_active_ms=sum(r['elapsed_ms'] for r in native['slots']),
                applied=applied, local_between_first_two=in_between,
                **{'pass': all(v['pass_gates'] and not v['contact'] for v in result.values())})


def run(args):
    plan = read(args.replay_from/'departure_plans.json')['independent']
    recorded = read(args.replay_from/'load_masks.json')
    if args.load_masks:
        masks = read(args.load_masks)
        if any(masks['windows'][name] != rows for name, rows in recorded['windows'].items()):
            raise ValueError('Frozen load masks differ in shared windows')
    else:
        source = args.trace_source or Path(recorded['source'])
        masks = load_windows(source, window_starts=WINDOW_STARTS)
    if masks['source_sha256'] != recorded['source_sha256']:
        raise ValueError('DPU trace does not match the frozen replay')
    available = {name: [row['slot'] for row in rows if not row['covering_call_ns']]
                 for name, rows in masks['windows'].items()}
    amplitudes, distances = ROUTES['independent']
    with scenario(PORT, args.output, amplitudes=amplitudes, distances=distances) as (client, gates):
        save(args.output/'load_masks.json', masks)
        probes = plan['probe_x_m']
        probe = fly(args, client, gates, args.output/'probe', amplitudes, distances, [8,16], probes, 'probe', 1, 11)
        if not (probe['pass'] and probe['adopted'] == 2 and probe['local_between_first_two'].get('Drone1', 0) > 0):
            raise RuntimeError('Fixed PS bridge must adopt both commands and locally control during the gap')
        if args.probe_only:
            return
        training = [fly(args, client, gates, args.output/f'tune_T{period}_c{model}', amplitudes, distances,
                        available['w01'], probes, 'fixed', period, model) for period, model in STATIC_CANDIDATES]
        viable = [r for r in training if r['pass'] and r['fleet_min_m'] is not None]
        if not viable:
            raise RuntimeError('No viable static candidate')
        best = max(viable, key=lambda r: (r['fleet_min_m'], -r['dpu_run_ms']))
        period, model = best['fixed_period_ms']//180, best['fixed_model']
        save(args.output/'static_choice.json', dict(period_slots=period, model=model,
                                                   tuning_window='w01', training=training))
        evaluation = {}
        for window in ('w09','w17'):
            for rep, order in enumerate((('fixed','adaptive'), ('adaptive','fixed')), 1):
                for policy in order:
                    name = f'{window}_r{rep}_{policy}'
                    evaluation[name] = fly(args, client, gates, args.output/name, amplitudes, distances,
                                           available[window], probes, policy, period, model)
                    save(args.output/'summary.json', evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-from', type=Path, required=True)
    loads = parser.add_mutually_exclusive_group()
    loads.add_argument('--trace-source', type=Path, help='Local path to the frozen historical DPU profile')
    loads.add_argument('--load-masks', type=Path, help='Frozen three-window output from this comparison')
    parser.add_argument('--binary', required=True, help='Fresh executable on the board')
    parser.add_argument('--package', default=BOARD_PACKAGE)
    parser.add_argument('--windows-ip', default='192.168.77.1')
    parser.add_argument('--board', default='axu3egb')
    parser.add_argument('--ssh-hop', default='vitis-vm')
    parser.add_argument('--probe-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
