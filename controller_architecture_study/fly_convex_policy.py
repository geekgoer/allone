"""Manual Play: verify the QP teacher, then a matching 300D or 237D DNN."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from config import load_calibration
from environment import Gate
from initial_state_generator import InitialState
from training_environment_generator import TrainingEnvironment
from gate_visualization import cleanup_generated_gates
from falcon_paper_fidelity.collector import _connect, _load_config, _resolve, run_episode
from experiments.falcon_paper_fidelity_acado_v3_continuous.demo_acado_five_gates import _crossing_results
from falcon_paper_fidelity.motion_primitive import QuinticMotionPrimitivePlanner
from .decision_dataset import ROOT, PLANNER_CONFIG, atomic_json, atomic_npz, sha256
from .decision_dataset_96k import route_for_group
from .convex_teacher import ConvexTeacher
from .convex_dataset import CONTRACT, VERSION, EXPERIMENT, specification
from .fly_decision_dnn import DecisionDNN, StoredRoutePlanner
from .eval_offline import load_checkpoint, load_normalization
from .convex_recovery_flight import PROFILES, PerturbedClient, RecoveryRecorder, recovery_metrics


def load_airsim_client():
    """Load only RPC packages, never prepend the entire second site's NumPy."""
    import airsim_interface

    if airsim_interface.airsim is not None:
        return
    source = Path("F:/Airsim/AirSim/PythonClient")
    libraries = source/".venv/Lib/site-packages"
    for name, root in (("tornado", libraries), ("msgpack", libraries),
                       ("msgpackrpc", libraries), ("airsim", source)):
        spec = importlib.machinery.PathFinder.find_spec(name, [str(root)])
        if spec is None:
            raise ImportError(f"missing AirSim dependency {name} under {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    airsim_interface.airsim = sys.modules["airsim"]


def verify_dataset(output, *, full=True):
    dataset = output/"dnn_dataset"
    manifest = json.loads((dataset/"split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("contract") != CONTRACT:
        raise ValueError("wrong dataset contract")
    if sha256(output/"manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("collection manifest changed")
    collection = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in collection["config"]["source_sha256"].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f"feature/teacher source changed: {name}")
    # Do not read test values for training/flight validation.
    for name in (("train.npz", "validation.npz", "normalization.json") if full else ("normalization.json",)):
        if sha256(dataset/name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"dataset artifact changed: {name}")
    return manifest


class ConvexDNN(DecisionDNN):
    def __init__(self, checkpoint, output, device):
        import torch

        config = json.loads((checkpoint.parent/"config.json").read_text(encoding="utf-8"))
        self.input_dim = config.get("input_dim")
        data_dir, contract = output/"dnn_dataset", CONTRACT
        if self.input_dim == 237:
            from .convex_recovery import verify_arm
            from .convex_features import CONTRACT as contract

            data_dir = ROOT/config["data_dir"]
            manifest = verify_arm(data_dir, output, full=False)
            if config.get("augmentation_manifest_sha256") != manifest["augmentation_manifest_sha256"]:
                raise ValueError("checkpoint augmentation provenance mismatch")
        else:
            manifest = verify_dataset(output, full=False)
        if (self.input_dim not in (237, 300) or config.get("input_contract") != contract
                or config.get("dataset_source_sha256") != manifest["source_manifest_sha256"]
                or config.get("dataset_file_sha256", {}).get("normalization.json") != manifest["artifact_sha256"]["normalization.json"]):
            raise ValueError("checkpoint/normalization/source mismatch")
        self.torch, self.device = torch, device
        self.model = load_checkpoint(checkpoint, device=device)
        self.norm = load_normalization(data_dir)
        self.config = SimpleNamespace(horizon=20, prediction_dt=.1)
        self.reset()
        with torch.inference_mode():
            for _ in range(10):
                self.model(torch.zeros((1, self.input_dim), device=device))
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def predict_raw(self, feature):
        if self.input_dim == 237:
            from .convex_features import project

            feature = project(feature)
        return super().predict_raw(feature)


def fly(args):
    import torch

    torch.set_num_threads(1)
    output = args.output.resolve()
    disturbance = getattr(args, "disturbance", "none")
    teacher_mode = args.checkpoint is None
    controller = ConvexTeacher() if teacher_mode else ConvexDNN(args.checkpoint.resolve(), output, args.device)
    if teacher_mode and disturbance == "none":
        _, layout = route_for_group(args.environment_id, args.route_index, 20, QuinticMotionPrimitivePlanner(PLANNER_CONFIG))
    else:
        collection = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        name = f"env{args.environment_id:02d}_route{args.route_index:02d}"
        path = output/"routes"/f"{name}.json"
        if not collection["complete"] or sha256(path) != collection["chunks"][name]["route_sha256"]:
            raise ValueError("unverified saved route")
        layout = json.loads(path.read_text(encoding="utf-8"))
    planner = StoredRoutePlanner(layout)
    environment = TrainingEnvironment(args.environment_id, layout["parent_route_generator_id"], 0.,
        f"convex_geometry_{layout['geometry_family']}",
        tuple(Gate(item["index"], item["center"], item["orientation"]) for item in layout["gates"]), reference_speed=5.)
    first = planner.reference_window_for_trajectory(planner.trajectory, elapsed=0., dt=.1, horizon=21)[0]
    initial = InitialState(0, first.position, first.orientation, np.zeros(3), np.zeros(3))
    config, config_path = _load_config(ROOT/"experiments/falcon_paper_fidelity_acado_v3_continuous/paper_core_config.json")
    config = deepcopy(config)
    config["vehicle_name"] = args.vehicle_name
    duration, dt = max(20., planner.trajectory.duration+2.), float(config["timing"]["dt"])
    config["timing"].update(steps_per_episode=int(np.ceil(duration/dt)), episode_timeout_seconds=duration+15.)
    config["gate"].update(spawn_in_simulation=True, keep_spawned_gates=args.keep_gates, preflight_hold_seconds=2.)
    config["trajectory"]["start_velocity_mode"] = "measured"
    config["bounds"]["position_max"][2] = -.3
    kick_time = getattr(args, "kick_time", 3.)
    if disturbance != "none" and (disturbance not in PROFILES or not np.isfinite(kick_time)
                                  or kick_time <= 0 or kick_time+3. > duration):
        raise ValueError("invalid disturbance or insufficient post-kick flight duration")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = output if teacher_mode else args.checkpoint.resolve().parent
    path = target/"flight"/f"env{args.environment_id:02d}_route{args.route_index:02d}_{stamp}.json"
    report = {"status": "FAIL", "controller": "convex_QP_teacher" if teacher_mode else f"convex_{controller.input_dim}D_DNN_only",
        "mpc_fallbacks": 0, "environment_id": args.environment_id, "route_index": args.route_index,
        "nominal_speed_mps": 5., "control_period_ms": dt*1000, "trajectory": "committed_global_C2_clock",
        "checkpoint": None if teacher_mode else str(args.checkpoint.resolve()), "error": None}
    report.update(disturbance=disturbance, report_path=str(path))
    if teacher_mode:
        report["teacher_config"] = specification(controller)
    started, episode, perturbed = time.perf_counter(), None, None
    try:
        load_airsim_client()
        client = _connect(config["airsim"])
        cleanup_generated_gates(client, prefix="FalconGate_")
        if disturbance != "none":
            controller = RecoveryRecorder(controller)
            client = perturbed = PerturbedClient(client, controller, profile=disturbance,
                heading_yaw=layout["heading_yaw"], dt=dt, vehicle_name=args.vehicle_name, kick_time=kick_time)
        print(f"{report['controller']}: 5m/s, T={dt*1000:.0f}ms; manual Play only; aerial rest start", flush=True)
        episode = run_episode(client, initial, environment, episode_id=-1, config=config,
            calibration=load_calibration(_resolve(config["calibration"], config_path.parent)), planner=planner, controller=controller)
    except KeyboardInterrupt:
        report["error"] = "interrupted by user"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if controller.records:
            arrays = {key: np.asarray([row[key] for row in controller.records]) for key in controller.records[0]}
            atomic_npz(path.with_suffix(".npz"), arrays)
            crossings = _crossing_results(arrays["position"], arrays["velocity"], environment, config["gate"])
            timing = arrays["solve_ms" if teacher_mode else "inference_ms"]
            displacement = float(np.linalg.norm(arrays["position"][min(int(5./dt), len(arrays["position"])-1)]-arrays["position"][0]))
            report.update(samples=len(controller.records), valid_crossings=sum(item[3] for item in crossings),
                gate_count=len(environment.gates), displacement_first_5s_m=displacement,
                mean_controller_ms=float(timing.mean()), p95_controller_ms=float(np.percentile(timing, 95)),
                max_position_error_m=float(np.linalg.norm(arrays["position"]-arrays["reference_position"], axis=1).max()),
                crossings=[{"gate": i, "valid": valid, "speed_mps": speed,
                    "lateral_m": None if cross is None else cross.lateral_offset,
                    "vertical_m": None if cross is None else cross.vertical_offset} for i, cross, speed, valid in crossings])
            if episode is not None:
                report["deadline_misses"] = int(episode.deadline_miss.sum())
            if report["error"] is None and episode is not None and len(crossings) == 5 and all(item[3] for item in crossings) and displacement > 1.:
                report["status"] = "PASS"
        if disturbance != "none":
            report["five_gate_flight_passed"] = report["status"] == "PASS"
            report["recovery"] = recovery_metrics(controller.records, None if perturbed is None else perturbed.event, dt)
            if not report["recovery"]["recovered"]:
                report["status"] = "FAIL"
                if report["error"] is None:
                    report["error"] = report["recovery"]["reason"]
        report["elapsed_seconds"] = time.perf_counter()-started
        atomic_json(path, report)
        # Never authorize new collection with a failed or interrupted flight.
        if teacher_mode and disturbance == "none":
            atomic_json(output/"teacher_acceptance.json", report)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Report: {path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="omit for QP teacher; supply for pure DNN")
    parser.add_argument("--output", type=Path, default=EXPERIMENT)
    parser.add_argument("--environment-id", type=int, choices=range(60), default=48)
    parser.add_argument("--route-index", type=int, choices=range(20), default=0)
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--disturbance", choices=("none", *PROFILES), default="none")
    parser.add_argument("--kick-time", type=float, default=3., help="committed trajectory time, not wall-clock time")
    args = parser.parse_args()
    if fly(args)["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
