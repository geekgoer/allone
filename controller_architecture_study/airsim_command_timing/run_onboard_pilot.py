"""Two AirSim plants, same pilot, a controlled gap in the central command path."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
from pathlib import Path
from threading import Event
from time import perf_counter, sleep

import numpy as np

from airsim_interface import require_airsim, vehicle_local_to_world_offset
from environment import Gate
from gate_route import segment_gate_crossing
from gate_visualization import spawn_gate_sequences
from .probe_blocks import LocalRpc


VEHICLES = {'Drone1': .65, 'Drone2': -.65}
DISTANCES = (3., 6., 9.)
Z = -3.
APERTURE = 2.6
BODY = .525
GAP_S = .18


def feed_central(port, vehicle, inject, stop):
    incident = None
    with LocalRpc(port) as rpc:
        seq, next_release = 0, perf_counter()
        while not stop.wait(max(0., next_release - perf_counter())):
            next_release = perf_counter() + .020
            seq += 1
            sampled = rpc.call('falconTraceClock')
            pose = rpc.call('getMultirotorState', vehicle)['kinematics_estimated']['position']
            xyz = [pose[f'{axis}_val'] for axis in 'xyz']
            if inject and incident is None and xyz[0] >= 4.7:
                start = rpc.call('falconTraceClock')
                if stop.wait(GAP_S):
                    break
                end = rpc.call('falconTraceClock')
                accepted = rpc.call('falconPilotSubmit', vehicle, seq, sampled, *xyz)
                incident = dict(start_ns=start, end_ns=end, late_accepted=accepted, parked_x_m=xyz[0])
            else:
                rpc.call('falconPilotSubmit', vehicle, seq, sampled, *xyz)
    return incident


def metrics(trace, gates, hit, incident):
    adopted = [row for row in trace if row[4] in (1, 2, 3)]
    pairs = list(zip(adopted, adopted[1:]))
    gaps = [(b[0] - a[0]) / 1e6 for a, b in pairs]
    incident_gaps = [(b[0] - a[0]) / 1e6 for a, b in pairs if incident and
                     a[0] < incident['end_ns'] and b[0] > incident['start_ns']]
    crossing = []
    for gate in gates:
        for a, b in zip(trace, trace[1:]):
            event = segment_gate_crossing(a[1:4], b[1:4], gate)
            if event:
                crossing.append(dict(gate=gate.index, estimated_body_clearance_m=APERTURE / 2 - BODY -
                                     max(abs(event.lateral_offset), abs(event.vertical_offset))))
                break
    margins = [item['estimated_body_clearance_m'] for item in crossing]
    approach = [APERTURE / 2 - BODY - max(abs(row[2] - gate.center[1]), abs(row[3] - gate.center[2]))
                for gate in gates for row in trace if abs(row[1] - gate.center[0]) <= BODY + .075]
    return dict(pass_gates=len(crossing) == len(gates) and min(margins) > 0 and not hit,
                crossings=crossing, minimum_clearance_m=min(margins, default=None), contact=hit,
                minimum_approach_clearance_m=min(approach, default=None),
                local_updates=sum(row[4] == 1 for row in adopted),
                central_updates=sum(row[4] in (2, 3) for row in adopted),
                local_updates_during_gap=sum(row[4] == 1 and incident['start_ns'] <= row[0] <= incident['end_ns']
                                             for row in adopted) if incident else 0,
                maximum_gap_ms=max(gaps, default=None), incident_maximum_gap_ms=max(incident_gaps, default=None),
                maximum_telemetry_gap_ms=max(((b[0]-a[0])/1e6 for a, b in zip(trace, trace[1:])), default=None))


def trial(client, port, directory, gates, mode, inject, shared_feed=None, *, stagger_s=0., leading='Drone1'):
    directory.mkdir()
    client.reset()
    client.simPause(False)
    for v in VEHICLES:
        client.enableApiControl(True, vehicle_name=v)
        if not client.armDisarm(True, vehicle_name=v):
            raise RuntimeError(f'Could not arm {v}')
    for future in [client.takeoffAsync(timeout_sec=7, vehicle_name=v) for v in VEHICLES]:
        future.join()
    sleep(.35)
    previous = {v: client.simGetCollisionInfo(vehicle_name=v).time_stamp for v in VEHICLES}
    client.simPause(True)
    stop = Event()
    started = []
    with LocalRpc(port) as rpc, ThreadPoolExecutor(max_workers=2) as pool:
        try:
            order = [leading] + [v for v in VEHICLES if v != leading]
            for v in order:
                if stagger_s and started:
                    client.simPause(False)
                    sleep(stagger_s)
                rpc.call('falconPilotStart', v, mode, VEHICLES[v], Z)
                started.append(v)
            client.simPause(False)
            feeders = {}
            shared = pool.submit(shared_feed, stop) if shared_feed else None
            if mode != 1 and not shared:
                feeders = {v: pool.submit(feed_central, port, v, inject and v == 'Drone1', stop) for v in VEHICLES}
            sleep(9.3)
        finally:
            stop.set()
            client.simPause(True)
            traces = {v: rpc.call('falconPilotStop', v) for v in started}
    report = {}
    incidents = {v: future.result() for v, future in feeders.items()}
    if shared:
        incidents, requests = shared.result()
        (directory / 'board_requests.json').write_text(json.dumps(requests, indent=2), encoding='utf-8')
    for v, trace in traces.items():
        incident = incidents.get(v)
        contact = client.simGetCollisionInfo(vehicle_name=v)
        hit = contact.object_name if contact.has_collided and contact.time_stamp != previous[v] else ''
        (directory / f'{v}.json').write_text(json.dumps(trace), encoding='utf-8')
        report[v] = dict(metrics(trace, gates[v], hit, incident), incident=incident)
        if hit:
            report[v]['contact_detail'] = dict(position=contact.position.to_numpy_array().tolist(),
                                               impact=contact.impact_point.to_numpy_array().tolist(),
                                               penetration_m=contact.penetration_depth)
    (directory / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


@contextmanager
def scenario(port, output):
    if port != 41592:
        raise ValueError('Use the isolated onboard pilot port 41592')
    with LocalRpc(port) as rpc:
        rpc.call('falconTraceClock')
    airsim = require_airsim()
    client = airsim.MultirotorClient(ip='127.0.0.1', port=port, timeout_value=8)
    if set(client.listVehicles()) != set(VEHICLES):
        raise RuntimeError('Expected the isolated two-vehicle scene')
    output.mkdir(parents=True)
    gates = {v: [Gate(i, np.array([x, amplitude * math.sin(.6 * x), Z]), np.array([1., 0., 0., 0.]))
                 for i, x in enumerate(DISTANCES, 1)] for v, amplitude in VEHICLES.items()}
    offsets = {v: vehicle_local_to_world_offset(client, v) for v in VEHICLES}
    scene = spawn_gate_sequences(client, gates, width=APERTURE + .15, height=APERTURE + .15, thickness=.15,
                                 object_prefix='FalconOnboardPilot_', vehicle_world_offsets=offsets)
    (output / 'scene.json').write_text(json.dumps(dict(offsets={v: d.tolist() for v, d in offsets.items()},
        bars={obj.name: dict(requested=obj.position.tolist(), scale=obj.scale.tolist(),
                       actual=client.simGetObjectPose(obj.name).position.to_numpy_array().tolist())
              for obj in scene.objects}), indent=2), encoding='utf-8')
    try:
        yield client, gates
    finally:
        client.simPause(False)
        client.reset()
        scene.destroy()


def run(port, output):
    reports = {}
    with scenario(port, output) as (client, gates):
        for name, mode, inject in [('local_normal', 1, False), ('central_normal', 2, False),
                                   ('central_gap', 2, True), ('hybrid_gap', 3, True)]:
            result = trial(client, port, output / name, gates, mode, inject)
            reports[name] = result
            print(name, [(v, r['pass_gates'], r['incident_maximum_gap_ms']) for v, r in result.items()], flush=True)
            if (not inject and not all(r['pass_gates'] for r in result.values())) or (
                    inject and (result['Drone1']['incident'] is None or result['Drone1']['incident']['late_accepted'])):
                break
        (output / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=41592)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.port, args.output)
    if len(result) != 4 or not all(r['pass_gates'] for r in result['hybrid_gap'].values()):
        raise SystemExit('Pilot did not satisfy the normal-flight and fallback checks')
    central = result['central_gap']['Drone1']
    hybrid = result['hybrid_gap']['Drone1']
    gaps = (hybrid['incident_maximum_gap_ms'], central['incident_maximum_gap_ms'])
    if (None in gaps or not hybrid['local_updates_during_gap'] or gaps[0] >= gaps[1]):
        raise SystemExit('Pilot did not establish a shorter command gap with fallback')


if __name__ == '__main__':
    main()
