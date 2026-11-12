"""Independent fixed-rate evaluation; never feeds state into control/planning."""
import threading
import time

import numpy as np

from .dynamic_probe_common import describe
from .fast_convex_preprocess import FastReferenceWindow
from .safety_comparison_core import paper_score


def observation(raw, ref, elapsed, sample_s):
    q, qr = np.asarray(raw.quaternion), np.asarray(ref.orientation)
    if np.dot(q, qr) < 0.:
        qr = -qr
    error = np.r_[raw.position-ref.position, raw.velocity-ref.velocity, q-qr]
    return dict(t=elapsed, sample_s=sample_s, sim_ns=int(raw.timestamp_ns or 0),
        position=raw.position.tolist(), velocity=raw.velocity.tolist(), quaternion=q.tolist(),
        reference_position=ref.position.tolist(), reference_velocity=ref.velocity.tolist(),
        reference_quaternion=qr.tolist(), state_cost=float(error@error))


class Evaluator:
    def __init__(self, host, name, planner, initial, origin, duration, period_ms):
        from airsim_interface import AirSimDrone, require_airsim
        self.client = require_airsim().MultirotorClient(ip=host, timeout_value=2.)
        self.client.ping()
        self.drone = AirSimDrone(self.client, name)
        self.window = FastReferenceWindow(planner.trajectory, planner.config)
        self.duration, self.period, self.origin = duration, period_ms/1000, origin
        self.trace = [observation(initial, self.window.window(0.)[0], 0., origin)]
        self.stop, self.error, self.rpc_ms = threading.Event(), None, []
        self.thread = threading.Thread(target=self.run, name=f'evaluate-{name}')

    def run(self):
        due = self.origin+self.period
        try:
            while not self.stop.wait(max(0., min(due, self.origin+self.duration)-time.perf_counter())):
                started = time.perf_counter()
                raw = self.drone.get_raw_state()
                self.rpc_ms.append(1000*(time.perf_counter()-started))
                elapsed = started-self.origin
                self.trace.append(observation(raw, self.window.window(elapsed)[0], elapsed, started))
                if elapsed >= self.duration:
                    break
                # No catch-up burst after a delayed RPC.
                due = self.origin+(int(elapsed/self.period)+1)*self.period
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def report(self):
        times = np.array([r['t'] for r in self.trace])
        gap = float(np.diff(times).max()*1000) if len(times) > 1 else None
        ratio = ((self.trace[-1]['sim_ns']-self.trace[0]['sim_ns'])/1e9/times[-1]
                 if len(times) > 1 and times[-1] > 0. else None)
        valid = (self.duration >= .4 and self.error is None and gap is not None and gap <= max(100., 3*self.period*1000)
                 and ratio is not None and .95 <= ratio <= 1.05)
        score = paper_score(self.trace, self.duration) if valid else None
        return dict(score=score, error=self.error, requested_period_ms=self.period*1000,
                    max_observation_gap_ms=gap, sim_wall_ratio=ratio, samples=len(times),
                    rpc_ms=describe(self.rpc_ms), valid=score is not None and score['J'] is not None)
