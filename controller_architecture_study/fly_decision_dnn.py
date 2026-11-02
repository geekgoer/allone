"""Manual AirSim flight for the 96k 300D policy, with a saved global C2 route."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from config import load_calibration
from environment import Gate
from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results
from falcon_paper_fidelity.collector import _connect, _load_config, _resolve, run_episode
from falcon_paper_fidelity.motion_primitive import (
    ContinuousQuinticTrajectory, QuinticMotionPrimitivePlanner, QuinticPolynomial, TimedQuinticSegment,
)
from gate_visualization import cleanup_generated_gates
from initial_state_generator import InitialState
from training_environment_generator import TrainingEnvironment
from .decision_dataset import CONTRACT, PLANNER_CONFIG, ROOT, atomic_json, atomic_npz, build_features, sha256
from .decision_dataset_96k import EXPERIMENT, VERSION
from .eval_offline import load_checkpoint, load_normalization


class StoredRoutePlanner(QuinticMotionPrimitivePlanner):
    def __init__(self, layout):
        super().__init__(PLANNER_CONFIG)
        segments = []
        for item in layout["segments"]:
            c = np.asarray(item["coefficients"], dtype=float)
            t = float(item["duration"])
            if c.shape != (3, 6) or not np.isfinite(c).all() or not np.isfinite(t) or t <= 0.:
                raise ValueError("invalid saved polynomial")
            end_p = c @ np.array([1., t, t**2, t**3, t**4, t**5])
            end_v = c[:, 1:] @ np.array([1., 2*t, 3*t**2, 4*t**3, 5*t**4])
            end_a = c[:, 2:] @ np.array([2., 6*t, 12*t**2, 20*t**3])
            primitive = QuinticPolynomial(c[:, 0], c[:, 1], 2*c[:, 2], end_p, end_v, end_a, t)
            primitive.coefficients = c.copy()  # Replay exact saved coefficients, without refitting.
            segments.append(TimedQuinticSegment(primitive, np.asarray(item["heading_orientation"]),
                                                float(item["start_time"])))
        self.trajectory = ContinuousQuinticTrajectory(segments)
        if not np.isclose(self.trajectory.duration, layout["duration"]):
            raise ValueError("saved trajectory duration mismatch")

    def plan_continuous_route(self, *args, **kwargs):
        # Feedback perturbations do not replan the path or reset its global clock.
        return self.trajectory


class DecisionDNN:
    def __init__(self, checkpoint, data_dir, device):
        import torch

        manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
        normalization = json.loads((data_dir / "normalization.json").read_text(encoding="utf-8"))
        config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
        if (manifest.get("version") != VERSION or manifest["contract"] != CONTRACT
                or normalization.get("input_contract") != CONTRACT or config.get("input_dim") != 300
                or config.get("dataset_source_sha256") != manifest["source_manifest_sha256"]):
            raise ValueError("model/data provenance or 300D input contract mismatch")
        self.torch, self.device = torch, device
        self.model = load_checkpoint(checkpoint, device=device)
        self.norm = load_normalization(data_dir)
        if len(self.norm["x_mean"]) != 300:
            raise ValueError("this flight entry requires 300D normalization")
        self.config = SimpleNamespace(horizon=20, prediction_dt=.1)
        self.reset()
        # Warm up before unpausing AirSim, so first-call setup is not a flight delay.
        with torch.inference_mode():
            for _ in range(10):
                self.model(torch.zeros((1, 300), device=device))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def reset(self):
        self.records = []

    def predict_raw(self, feature):
        x = (feature - self.norm["x_mean"]) / self.norm["x_std"]
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(x[None, :]).to(self.device)).cpu().numpy()[0]
        action = output * self.norm["u_std"] + self.norm["u_mean"]
        if action.shape != (4,) or not np.isfinite(action).all():
            raise RuntimeError("DNN returned an invalid action")
        return action.astype(float)

    def solve(self, state, references):
        started = time.perf_counter()
        feature = build_features(state, references)
        raw = self.predict_raw(feature)
        action = np.clip(raw, [2., -3., -3., -2.], [20., 3., 3., 2.])
        self.records.append({"X": feature, "raw_U": raw, "U": action,
            "position": state.position.copy(), "velocity": state.velocity.copy(),
            "reference_position": references[0].position.copy(),
            "inference_ms": (time.perf_counter() - started) * 1000,
            "clipped": bool(np.any(raw != action))})
        return action


def load_route(environment_id, route_index):
    name = f"env{environment_id:02d}_route{route_index:02d}"
    path = EXPERIMENT / "routes" / f"{name}.json"
    manifest = json.loads((EXPERIMENT / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["config"]["version"] != VERSION:
        raise ValueError("complete 96k collection required")
    if sha256(path) != manifest["chunks"][name]["route_sha256"]:
        raise ValueError("saved route changed since collection")
    layout = json.loads(path.read_text(encoding="utf-8"))
    if layout["environment_id"] != environment_id:
        raise ValueError("saved route environment mismatch")
    environment = TrainingEnvironment(environment_id, layout["parent_route_generator_id"], 0.,
        f"decision_geometry_{layout['geometry_family']}",
        tuple(Gate(item["index"], item["center"], item["orientation"]) for item in layout["gates"]),
        reference_speed=5.)
    return layout, environment


def fly(args):
    import torch

    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = args.checkpoint.resolve()
    controller = DecisionDNN(checkpoint, EXPERIMENT / "dnn_dataset", args.device)
    layout, environment = load_route(args.environment_id, args.route_index)
    planner = StoredRoutePlanner(layout)
    first = planner.reference_window_for_trajectory(planner.trajectory, elapsed=0., dt=.1, horizon=21)[0]
    initial = InitialState(0, first.position, first.orientation, first.velocity, np.zeros(3))
    config, config_path = _load_config(ROOT / "experiments/falcon_paper_fidelity_acado_v3_continuous/paper_core_config.json")
    config = deepcopy(config)
    config["vehicle_name"] = args.vehicle_name
    duration = args.duration if args.duration is not None else max(20., planner.trajectory.duration + 2.)
    if not np.isfinite(duration) or duration <= 0.:
        raise ValueError("duration must be positive and finite")
    dt = float(config["timing"]["dt"])
    config["timing"].update(steps_per_episode=int(np.ceil(duration / dt)),
                            episode_timeout_seconds=duration + 15.)
    config["gate"].update(spawn_in_simulation=True, keep_spawned_gates=args.keep_gates,
                           preflight_hold_seconds=2.)
    config["trajectory"]["start_velocity_mode"] = "measured"
    config["bounds"]["position_max"][2] = -.3  # Stop descent before entering the ground.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = checkpoint.parent / "flight" / f"env{args.environment_id:02d}_route{args.route_index:02d}_{stamp}.json"
    report = {"status": "FAIL", "controller": "300D_DNN_only", "mpc_fallbacks": 0,
        "checkpoint": str(checkpoint), "input_contract": CONTRACT, "environment_id": args.environment_id,
        "route_index": args.route_index, "route_id": layout["route_id"],
        "trajectory": "saved_committed_global_C2_clock", "nominal_speed_mps": 5.,
        "control_period_ms": dt * 1000, "error": None}
    started = time.perf_counter()
    episode = None
    try:
        # Manual Play session only; never auto-launch another Unreal process.
        client = _connect(config["airsim"])
        cleanup_generated_gates(client, prefix="FalconGate_")
        print(f"DNN ONLY: {checkpoint.parent.name}, environment={args.environment_id}, "
              f"route={args.route_index}, geometry={layout['geometry_family']}, 5m/s, T={dt*1000:.0f}ms", flush=True)
        print("The vehicle is placed at the saved aerial start (-4m), not a ground takeoff.", flush=True)
        episode = run_episode(client, initial, environment, episode_id=-1, config=config,
            calibration=load_calibration(_resolve(config["calibration"], config_path.parent)),
            planner=planner, controller=controller)
    except KeyboardInterrupt:
        report["error"] = "KeyboardInterrupt: flight interrupted by user"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rows = controller.records
        if rows:
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            trace = report_path.with_suffix(".npz")
            atomic_npz(trace, arrays)
            crossings = _crossing_results(arrays["position"], arrays["velocity"], environment, config["gate"])
            report.update(samples=len(rows), valid_crossings=sum(item[3] for item in crossings), gate_count=5,
                trace=str(trace), clipped_cycles=int(arrays["clipped"].sum()),
                mean_inference_ms=float(arrays["inference_ms"].mean()),
                p95_inference_ms=float(np.percentile(arrays["inference_ms"], 95)),
                max_position_error_m=float(np.max(np.linalg.norm(arrays["position"] - arrays["reference_position"], axis=1))),
                crossings=[{"gate": index, "valid": valid, "speed_mps": speed,
                    "lateral_m": None if crossing is None else crossing.lateral_offset,
                    "vertical_m": None if crossing is None else crossing.vertical_offset}
                    for index, crossing, speed, valid in crossings])
            if episode is not None:
                report["deadline_misses"] = int(episode.deadline_miss.sum())
            if report["error"] is None and episode is not None and all(item[3] for item in crossings):
                report["status"] = "PASS"
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Report: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-id", type=int, choices=range(60), default=48)
    parser.add_argument("--route-index", type=int, choices=range(20), default=0)
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--keep-gates", action="store_true")
    args = parser.parse_args()
    if fly(args)["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
