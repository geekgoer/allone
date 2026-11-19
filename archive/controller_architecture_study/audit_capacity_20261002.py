"""Read-only CPU diagnostics of the existing SCENIC-named 12-D runs.

Does not train, connect to AirSim, or change datasets/checkpoints. Only the
explicit diagnostic report is written. The test split is not loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .eval_offline import load_checkpoint, load_normalization, load_split
from mpc_controller import MPCController
from track_p_12d_contract import build_track_p_teacher_query


ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "controller_architecture_study"
DATA = ROOT / "experiments/paper_track_p_12d_5ms_v1/dnn_dataset"


def query(controller, x, gate=None):
    return build_track_p_teacher_query(
        gate_position=np.zeros(3) if gate is None else gate,
        gate_orientation=[1, 0, 0, 0],
        gate_position_error=x[:3], gate_attitude_error=x[3:6],
        velocity=x[6:9], acceleration=x[9:12],
        dt=controller.dt, horizon=controller.horizon, reference_speed=5.0,
    )


def preclip_acceleration(controller, x):
    q = query(controller, x)
    refs = np.concatenate([np.r_[r.position, r.velocity] for r in q.reference_window])
    state = np.r_[q.current_position, x[6:9]]
    return (controller._reference_tracking_gain @
            (refs - controller._state_prediction @ state))[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    import torch

    torch.set_num_threads(4)
    rng = np.random.default_rng(20261002)
    norm = load_normalization(DATA)
    subsets = {}
    for split in ("train", "validation"):
        x, u = load_split(DATA, split)
        ids = rng.choice(len(x), min(args.samples, len(x)), replace=False)
        subsets[split] = (x[ids], u[ids])

    controller = MPCController(dt=0.05, horizon=20, max_horizontal_accel=6,
                               max_thrust_accel=18, max_body_rate=3)
    intercept = preclip_acceleration(controller, np.zeros(12))
    coefficients = np.column_stack([
        preclip_acceleration(controller, e) - intercept for e in np.eye(12)
    ])
    differences = {k: [] for k in ("stored_label", "acceleration_change",
                                    "longitudinal_distance_change", "gate_translation")}
    for x, u in zip(*[a[:256] for a in subsets["train"]]):
        q = query(controller, x)
        original = controller.solve(q.teacher_state, q.reference_window)
        differences["stored_label"].append(float(np.max(np.abs(original - u))))
        for key in ("acceleration_change", "longitudinal_distance_change", "gate_translation"):
            altered = x.astype(np.float64).copy()
            gate = None
            if key == "acceleration_change":
                altered[9:12] = rng.uniform(-4, 4, size=3)
            elif key == "longitudinal_distance_change":
                altered[0] = rng.uniform(0.05, 16)
            else:
                gate = rng.uniform(-50, 50, size=3)
            q = query(controller, altered, gate)
            predicted = controller.solve(q.teacher_state, q.reference_window)
            differences[key].append(float(np.max(np.abs(predicted - original))))
    report = {
        "device": "cpu", "test_split_used": False, "subset_seed": 20261002,
        "teacher_replay_samples": len(differences["stored_label"]),
        "teacher_max_abs_differences": {k: max(v) for k, v in differences.items()},
        "preclip_acceleration_intercept": intercept.tolist(),
        "preclip_acceleration_coefficients_3_by_12": coefficients.tolist(),
        "models": [],
    }
    folders = [
        STUDY / "results/plain_mlp_h4_w128_seed0",
        STUDY / "results/scenic_table3_v1/plain_mlp_h4_w1024_seed0",
        STUDY / "results/scenic_table3_v1/plain_mlp_h8_w256_seed0",
        STUDY / "results/scenic_table3_v1/plain_mlp_h12_w1024_seed0",
    ]
    print(json.dumps({k: v for k, v in report.items() if k != "models"}), flush=True)
    for folder in folders:
        model = load_checkpoint(folder / "best_model.pt", device="cpu")
        metrics = {"model": folder.name, "checkpoint": str(folder / "best_model.pt")}
        with torch.inference_mode():
            for split, (x, u) in subsets.items():
                predictions = []
                for start in range(0, len(x), 512):
                    batch = (x[start:start+512] - norm["x_mean"]) / norm["x_std"]
                    predictions.append(model(torch.from_numpy(batch)).numpy())
                error = np.concatenate(predictions) - (u - norm["u_mean"]) / norm["u_std"]
                metrics[split] = {"samples": len(x), "normalized_mse": float(np.mean(error**2))}
        log = list(csv.DictReader((folder / "train_log.csv").open(encoding="utf-8")))
        metrics["best_epoch"] = int(min(log, key=lambda r: float(r["validation_normalized_mse"]))["epoch"])
        metrics["epochs_completed"] = len(log)
        report["models"].append(metrics)
        print(json.dumps(metrics), flush=True)
        del model
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
