"""Audit a short-rollout teacher dataset before train/validation/test splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.recovery_dataset import CONTROL_BUCKET_NAMES, classify_control_bucket
from falcon_paper_fidelity.short_rollout_dataset import FAMILY_NAMES


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "short_rollout_120k_config.json")
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=here / "short_rollout_120k_audit.json")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    teacher_root = (
        args.teacher_root.resolve()
        if args.teacher_root is not None
        else (config_path.parent / config["dataset"]["root"]).resolve()
    )
    manifest = json.loads((teacher_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("teacher dataset is not complete")
    with np.load(teacher_root / manifest["final_dataset"], allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "X", "U", "sample_id", "query_group_id", "rollout_step", "sample_family",
        "control_bucket", "position", "quaternion", "velocity", "acceleration",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise RuntimeError(f"dataset is missing arrays: {missing}")
    x, u = arrays["X"], arrays["U"]
    if x.shape != (int(config["dataset"]["sample_count"]), 12) or u.shape != (len(x), 4):
        raise RuntimeError(f"unexpected X/U shapes: {x.shape}, {u.shape}")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise RuntimeError("X or U contains NaN/Inf")

    rollout_steps = int(config["dataset"]["rollout_steps"])
    group_ids = np.asarray(arrays["query_group_id"], dtype=np.int64)
    steps = np.asarray(arrays["rollout_step"], dtype=np.int64)
    unique_groups, counts = np.unique(group_ids, return_counts=True)
    bad_group_sizes = int(np.count_nonzero(counts != rollout_steps))
    bad_step_sets = 0
    for group in unique_groups:
        values = np.sort(steps[group_ids == group])
        bad_step_sets += int(not np.array_equal(values, np.arange(rollout_steps)))
    if bad_group_sizes or bad_step_sets:
        raise RuntimeError(
            f"invalid rollout grouping: bad_sizes={bad_group_sizes}, bad_step_sets={bad_step_sets}"
        )

    mpc = config["mpc"]
    lower = np.array([mpc["min_thrust"], -mpc["max_body_rate_xy"], -mpc["max_body_rate_xy"], -mpc["max_body_rate_z"]])
    upper = np.array([mpc["max_thrust"], mpc["max_body_rate_xy"], mpc["max_body_rate_xy"], mpc["max_body_rate_z"]])
    expected_buckets = np.asarray(
        [
            classify_control_bucket(
                row, lower, upper,
                saturation_tolerance=float(config["sampling"]["saturation_tolerance"]),
                near_limit_fraction=float(config["sampling"]["near_limit_fraction"]),
            )
            for row in u
        ],
        dtype=np.int8,
    )
    bucket = np.asarray(arrays["control_bucket"], dtype=np.int8)
    if not np.array_equal(bucket, expected_buckets):
        raise RuntimeError("stored control buckets disagree with recomputed buckets")
    bucket_counts = np.bincount(bucket.astype(np.int64), minlength=3)
    saturated_fraction = float(bucket_counts[2] / len(x))
    near_fraction = float(bucket_counts[1] / len(x))
    if saturated_fraction > float(config["sampling"]["maximum_saturated_fraction"]) + 1e-12:
        raise RuntimeError("saturated fraction exceeds configured limit")
    if near_fraction > float(config["sampling"]["maximum_near_limit_fraction"]) + 1e-12:
        raise RuntimeError("near-limit fraction exceeds configured limit")

    family = np.asarray(arrays["sample_family"], dtype=np.int8)
    family_counts = {
        FAMILY_NAMES[index]: int(np.count_nonzero(family == index))
        for index in range(len(FAMILY_NAMES))
    }
    speed = np.linalg.norm(np.asarray(arrays["velocity"], dtype=np.float64), axis=1)
    rollout_position_delta = []
    rollout_speed_delta = []
    for group in unique_groups:
        mask = group_ids == group
        order = np.argsort(steps[mask])
        positions = arrays["position"][mask][order]
        velocities = arrays["velocity"][mask][order]
        rollout_position_delta.extend(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        rollout_speed_delta.extend(np.linalg.norm(np.diff(velocities, axis=0), axis=1))
    if not rollout_position_delta or not rollout_speed_delta:
        raise RuntimeError("no rollout transitions found")

    report = {
        "status": "valid",
        "teacher_root": str(teacher_root),
        "final_dataset": str((teacher_root / manifest["final_dataset"]).resolve()),
        "samples": int(len(x)),
        "X_shape": list(x.shape),
        "U_shape": list(u.shape),
        "rollouts": int(len(unique_groups)),
        "rollout_steps": rollout_steps,
        "control_dt_seconds": float(config["timing"]["control_dt"]),
        "reference_speed_mps": float(config["planner"]["nominal_speed"]),
        "horizon": int(config["mpc"]["horizon"]),
        "prediction_dt_seconds": float(config["mpc"]["prediction_dt"]),
        "family_counts": family_counts,
        "control_bucket_counts": {
            CONTROL_BUCKET_NAMES[index]: int(bucket_counts[index]) for index in range(3)
        },
        "saturated_fraction": saturated_fraction,
        "near_limit_fraction": near_fraction,
        "limits": {"maximum_saturated_fraction": float(config["sampling"]["maximum_saturated_fraction"]), "maximum_near_limit_fraction": float(config["sampling"]["maximum_near_limit_fraction"])},
        "speed_mps": _quantiles(speed),
        "position_error_x": _quantiles(x[:, 0]),
        "control_per_output": {
            name: _quantiles(u[:, index])
            for index, name in enumerate(("thrust", "body_rate_x", "body_rate_y", "body_rate_z"))
        },
        "transition_position_delta_m": _quantiles(np.asarray(rollout_position_delta)),
        "transition_velocity_delta_mps": _quantiles(np.asarray(rollout_speed_delta)),
        "mean_mpc_solve_ms": float(np.mean(arrays["mpc_solve_time"]) * 1e3),
        "warnings": [],
    }
    if report["speed_mps"]["p50"] < 8.0:
        report["warnings"].append("median state speed is unexpectedly low for a 10 m/s experiment")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
