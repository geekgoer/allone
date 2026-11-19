"""Validate the independent Track-P 12-D dataset and recompute MPC labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mpc_controller import MPCController
from track_p_12d_contract import TRACK_P_12D_INPUT_FEATURES, solve_track_p_teacher
from training_environment_generator import load_training_environments


def validate(root: Path, *, label_checks: int) -> dict[str, object]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"dataset is not completed: {manifest.get('status')!r}")
    if int(manifest.get("input_dimensions", -1)) != 12:
        raise ValueError("manifest does not describe a 12-D dataset")
    if manifest.get("input_features") != list(TRACK_P_12D_INPUT_FEATURES):
        raise ValueError("manifest input features do not match the Track-P contract")
    config = json.loads((root / "collection_config.json").read_text(encoding="utf-8"))
    final_path = root / Path(str(manifest["final_dataset"]))
    with np.load(final_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "X",
        "U",
        "state_X",
        "teacher_state_X",
        "current_position",
        "gate_position",
        "gate_orientation",
        "reference_first_position",
        "reference_last_position",
        "reference_speed",
        "environment_id",
        "target_gate_id",
        "sample_family",
        "data_source",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"dataset is missing arrays: {missing}")
    rows = int(arrays["X"].shape[0])
    expected_rows = int(manifest["environment_count"]) * int(manifest["samples_per_environment"])
    if arrays["X"].shape != (expected_rows, 12):
        raise ValueError(f"unexpected X shape: {arrays['X'].shape}")
    if arrays["U"].shape != (expected_rows, 4):
        raise ValueError(f"unexpected U shape: {arrays['U'].shape}")
    for key, value in arrays.items():
        if value.ndim >= 1 and value.shape[0] == rows and np.issubdtype(value.dtype, np.number):
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN or Inf")
    np.testing.assert_allclose(arrays["X"], arrays["state_X"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        arrays["X"][:, 0:3],
        arrays["gate_position"] - arrays["current_position"],
        rtol=0.0,
        atol=2e-6,
    )
    if not np.all(arrays["data_source"] == 4):
        raise ValueError("Track-P independent queries must use data_source=4")
    if np.any(arrays["X"][:, 0] <= 0.0):
        raise ValueError("all current gates must be ahead of the sampled vehicle")
    reference_speeds = np.unique(arrays["reference_speed"])
    if reference_speeds.size != 1:
        raise ValueError(
            "12-D data cannot hide multiple reference speeds; expected exactly one"
        )

    environments = {
        item.environment_id: item
        for item in load_training_environments(root / "environments.json")
    }
    mpc_config = config["mpc"]
    dt = float(config["timing"]["dt"])
    controller = MPCController(
        dt=dt,
        horizon=int(mpc_config["horizon"]),
        max_horizontal_accel=float(mpc_config.get("max_horizontal_accel", 4.0)),
        max_thrust_accel=float(mpc_config.get("max_thrust_accel", 18.0)),
        max_body_rate=float(mpc_config.get("max_body_rate", 3.0)),
    )
    check_count = min(max(0, int(label_checks)), rows)
    indices = (
        np.linspace(0, rows - 1, num=check_count, dtype=np.int64)
        if check_count
        else np.empty(0, dtype=np.int64)
    )
    max_label_error = 0.0
    max_input_error = 0.0
    max_teacher_state_error = 0.0
    max_reference_error = 0.0
    for index in indices:
        environment_id = int(arrays["environment_id"][index])
        gate_index = int(arrays["target_gate_id"][index]) - 1
        gate = environments[environment_id].gates[gate_index]
        np.testing.assert_allclose(arrays["gate_position"][index], gate.center, atol=1e-6)
        control, query = solve_track_p_teacher(
            controller,
            gate_position=gate.center,
            gate_orientation=gate.orientation,
            gate_position_error=arrays["X"][index, 0:3],
            gate_attitude_error=arrays["X"][index, 3:6],
            velocity=arrays["X"][index, 6:9],
            acceleration=arrays["X"][index, 9:12],
            dt=dt,
            horizon=controller.horizon,
            reference_speed=float(arrays["reference_speed"][index]),
        )
        max_input_error = max(
            max_input_error,
            float(np.max(np.abs(query.policy_input - arrays["X"][index]))),
        )
        max_teacher_state_error = max(
            max_teacher_state_error,
            float(np.max(np.abs(query.teacher_state - arrays["teacher_state_X"][index]))),
        )
        max_reference_error = max(
            max_reference_error,
            float(
                max(
                    np.max(
                        np.abs(
                            query.reference_window[0].position
                            - arrays["reference_first_position"][index]
                        )
                    ),
                    np.max(
                        np.abs(
                            query.reference_window[-1].position
                            - arrays["reference_last_position"][index]
                        )
                    ),
                )
            ),
        )
        max_label_error = max(
            max_label_error,
            float(np.max(np.abs(control - arrays["U"][index]))),
        )
    families, family_counts = np.unique(arrays["sample_family"], return_counts=True)
    gates, gate_counts = np.unique(arrays["target_gate_id"], return_counts=True)
    unique_x = np.unique(arrays["X"], axis=0).shape[0]
    rate_limit = float(mpc_config.get("max_body_rate", 3.0))
    thrust_limit = float(mpc_config.get("max_thrust_accel", 18.0))
    x_std = np.std(arrays["X"], axis=0)
    u_std = np.std(arrays["U"], axis=0)
    if np.any(x_std <= 1e-8) or np.any(u_std <= 1e-8):
        raise ValueError("one or more input/output dimensions are effectively constant")
    family_names = list(manifest["sample_family_names"])
    result = {
        "status": "valid",
        "root": str(root),
        "final_dataset": str(final_path),
        "strategy": manifest["strategy"],
        "input_contract": manifest["input_contract"],
        "samples": rows,
        "X_shape": list(arrays["X"].shape),
        "U_shape": list(arrays["U"].shape),
        "environments": int(np.unique(arrays["environment_id"]).size),
        "reference_speed_values": reference_speeds.astype(float).tolist(),
        "unique_input_rows": int(unique_x),
        "duplicate_input_fraction": float(1.0 - unique_x / max(rows, 1)),
        "X_std_per_dim": x_std.astype(float).tolist(),
        "U_std_per_dim": u_std.astype(float).tolist(),
        "sample_family_counts": {
            family_names[int(key)]: int(value)
            for key, value in zip(families, family_counts)
        },
        "target_gate_counts": {
            str(int(key)): int(value) for key, value in zip(gates, gate_counts)
        },
        "thrust_saturated_fraction": float(
            np.mean((arrays["U"][:, 0] <= 1e-6) | (arrays["U"][:, 0] >= thrust_limit - 1e-6))
        ),
        "any_body_rate_saturated_fraction": float(
            np.mean(np.any(np.abs(arrays["U"][:, 1:4]) >= rate_limit - 1e-6, axis=1))
        ),
        "label_checks": check_count,
        "max_recomputed_input_abs_error": max_input_error,
        "max_recomputed_teacher_state_abs_error": max_teacher_state_error,
        "max_recomputed_reference_abs_error": max_reference_error,
        "max_recomputed_label_abs_error": max_label_error,
        "measured_acceleration_used_by_current_teacher": bool(
            manifest["measured_acceleration_used_by_current_teacher"]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label-checks", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_root, label_checks=args.label_checks), indent=2))


if __name__ == "__main__":
    main()
