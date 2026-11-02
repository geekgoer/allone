"""Short CPU-only parity/timing check before the vectorized three-drone trial."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import numpy as np

from airsim_interface import RawDroneState
from .convex_features import encode
from .convex_teacher import ConvexTeacher
from .decision_dataset import atomic_json
from .dynamic_probe_common import ModelPool, describe, new_output, provenance
from .fast_convex_preprocess import FastReferenceWindow, encode as fast_encode
from .fly_decision_dnn import StoredRoutePlanner
from .probe_airsim_three import load_layout


def benchmark(planner, fast, state, clients, vectorized):
    barrier = threading.Barrier(clients)
    def actor(index):
        barrier.wait()
        rows = []
        for k in range(40):
            elapsed = ((k+index)*.13) % planner.trajectory.duration
            start = time.perf_counter()
            refs = fast.window(elapsed) if vectorized else planner.reference_window_for_trajectory(
                planner.trajectory, elapsed=elapsed, dt=.1, horizon=21)
            middle = time.perf_counter()
            (fast_encode if vectorized else encode)(state, refs)
            end = time.perf_counter()
            rows.append([(middle-start)*1000, (end-middle)*1000, (end-start)*1000])
        return rows
    with ThreadPoolExecutor(max_workers=clients) as executor:
        rows = np.asarray([row for group in executor.map(actor, range(clients)) for row in group])
    return {"clients": clients, "policy": "vectorized" if vectorized else "parallel",
            **{name: describe(rows[:, i]) for i, name in enumerate(("reference_ms", "feature_ms", "total_ms"))}}


def main():
    path = new_output("preprocess_check")
    evidence = provenance(["residual21x1024"])
    pool = ModelPool(["residual21x1024"], "cpu")
    rng = np.random.default_rng(20261004)
    maximum = {name: 0. for name in ("position", "velocity", "acceleration", "orientation", "body_rates")}
    old_features, new_features, cases = [], [], []
    teacher_delta = 0.
    teacher = ConvexTeacher()
    for environment in range(48, 54):  # Validation routes only; no Test split.
        layout, route_hash = load_layout(environment, 0)
        planner = StoredRoutePlanner(layout)
        fast = FastReferenceWindow(planner.trajectory, planner.config)
        boundaries = np.array([*planner.trajectory.boundary_times, planner.trajectory.duration])
        phases = np.unique(np.r_[0., boundaries-1e-9, boundaries, boundaries+1e-9,
            planner.trajectory.duration+1., rng.uniform(0., planner.trajectory.duration+2., 16)])
        for index, elapsed in enumerate(phases):
            refs = planner.reference_window_for_trajectory(planner.trajectory,
                elapsed=float(elapsed), dt=.1, horizon=21)
            batched = fast.window(float(elapsed))
            for name in maximum:
                left, right = np.array([getattr(r, name) for r in refs]), np.array([getattr(r, name) for r in batched])
                maximum[name] = max(maximum[name], float(np.max(abs(left-right))))
                np.testing.assert_allclose(right, left, rtol=1e-10,
                    atol=2e-8 if name == "body_rates" else 1e-10)
            q = refs[0].orientation * (-1. if index % 2 else 1.)
            state = RawDroneState(refs[0].position+rng.uniform(-1., 1., 3), q,
                refs[0].velocity+rng.uniform(-1., 1., 3), np.zeros(3), np.zeros(3))
            left, right = encode(state, refs), fast_encode(state, batched)
            np.testing.assert_allclose(right, left, rtol=0., atol=2e-6)
            old_features.append(left)
            new_features.append(right)
            cases.append({"environment_id": environment, "route_index": 0, "route_sha256": route_hash,
                          "elapsed": float(elapsed)})
            if index % 8 == 0:
                delta = float(np.max(abs(teacher.query(state, refs)[0]-teacher.query(state, batched)[0])))
                teacher_delta = max(teacher_delta, delta)
                if delta > 2e-5:
                    raise AssertionError("MPC action parity failed")
    old, new = np.stack(old_features), np.stack(new_features)
    normalized_delta = np.max(abs((new-old)/pool.norm["x_std"]))
    if normalized_delta > 2e-5:
        raise AssertionError("normalized input parity failed")
    chosen = np.linspace(0, len(old)-1, 24).astype(int)
    def predict(features):
        x = (features[chosen]-pool.norm["x_mean"])/pool.norm["x_std"]
        with pool.torch.inference_mode():
            y = pool.models["residual21x1024"](pool.torch.from_numpy(x)).numpy()
        return y*pool.norm["u_std"]+pool.norm["u_mean"]
    action_delta = float(np.max(abs(predict(old)-predict(new))))
    if action_delta > 2e-5:
        raise AssertionError("FP32 DNN action parity failed")
    layout, _ = load_layout(48, 0)
    planner = StoredRoutePlanner(layout)
    fast = FastReferenceWindow(planner.trajectory, planner.config)
    ref = fast.window(.5)[0]
    state = RawDroneState(ref.position, ref.orientation, ref.velocity, np.zeros(3), np.zeros(3))
    for _ in range(5):
        fast_encode(state, fast.window(.5))
        encode(state, planner.reference_window_for_trajectory(planner.trajectory, elapsed=.5, dt=.1, horizon=21))
    timings = [benchmark(planner, fast, state, n, vectorized)
               for n, vectorized in ((1, False), (1, True), (3, False), (3, True))]
    report = {"status": "PASS", "scope": "CPU-only numerical parity and preprocessing microbenchmark",
        "test_read": False, "airsim_connected": False, "cuda_inference_run": False,
        "provenance": evidence, "cases": cases, "reference_max_abs_delta": maximum,
        "feature_max_abs_delta": float(np.max(abs(new-old))),
        "normalized_feature_max_abs_delta": float(normalized_delta),
        "mpc_action_max_abs_delta": teacher_delta, "cpu_dnn_action_max_abs_delta": action_delta,
        "cpu_dnn_checked_samples": len(chosen), "timings": timings,
        "limitations": ["CPU microbenchmark is not the rendered AirSim end-to-end timing result.",
                        "Finite numerical probes, not an all-input equivalence proof or a deadline guarantee."]}
    atomic_json(path / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "provenance")}, indent=2))
    print(f"Report: {path / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
