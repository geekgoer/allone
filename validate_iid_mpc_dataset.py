"""Validate an independent path-aware MPC query dataset and sampled labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dnn_model_utils import PATH_AWARE_INPUT_FEATURES, PATH_CONTEXT_OFFSETS, build_policy_input
from generate_iid_mpc_dataset import _reference_window, _route_geometry
from mpc_controller import MPCController
from training_environment_generator import load_training_environments


def validate(root: Path, *, label_checks: int) -> dict[str, object]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"dataset is not completed: {manifest.get('status')!r}")
    config = json.loads((root / "collection_config.json").read_text(encoding="utf-8"))
    final_path = root / Path(str(manifest["final_dataset"]))
    with np.load(final_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "X", "U", "state_X", "path_context", "reference_speed", "environment_id",
        "target_gate_id", "route_progress", "sample_family", "data_source",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"dataset is missing arrays: {missing}")
    rows = int(arrays["X"].shape[0])
    expected_rows = int(manifest["environment_count"]) * int(manifest["samples_per_environment"])
    if arrays["X"].shape != (expected_rows, len(PATH_AWARE_INPUT_FEATURES)):
        raise ValueError(f"unexpected X shape: {arrays['X'].shape}")
    if arrays["U"].shape != (expected_rows, 4) or arrays["state_X"].shape != (expected_rows, 12):
        raise ValueError("unexpected U or state_X shape")
    if arrays["path_context"].shape != (expected_rows, len(PATH_CONTEXT_OFFSETS), 3):
        raise ValueError("unexpected path_context shape")
    for key, value in arrays.items():
        if value.ndim >= 1 and value.shape[0] == rows and np.issubdtype(value.dtype, np.number):
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN or Inf")
    np.testing.assert_allclose(arrays["X"][:, :12], arrays["state_X"], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        arrays["X"][:, 12:24], arrays["path_context"].reshape(rows, 12), rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(arrays["X"][:, 24], arrays["reference_speed"], rtol=0.0, atol=1e-6)
    if not np.all(arrays["data_source"] == 3):
        raise ValueError("independent query dataset must use data_source=3")

    environments = {item.environment_id: item for item in load_training_environments(root / "environments.json")}
    route_config = config["route"]
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
    indices = np.linspace(0, rows - 1, num=check_count, dtype=np.int64) if check_count else np.empty(0, dtype=np.int64)
    max_label_error = 0.0
    max_input_error = 0.0
    geometry_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for index in indices:
        environment_id = int(arrays["environment_id"][index])
        environment = environments[environment_id]
        geometry = geometry_cache.get(environment_id)
        if geometry is None:
            geometry = _route_geometry(
                environment,
                start_x=float(route_config["start_x"]),
                approach_distance=float(route_config["approach_distance"]),
                exit_distance=float(route_config["exit_distance"]),
            )
            geometry_cache[environment_id] = geometry
        waypoints, cumulative, _, total_length = geometry
        gate_index = int(arrays["target_gate_id"][index]) - 1
        progress = float(arrays["route_progress"][index]) * total_length
        window = _reference_window(
            waypoints,
            cumulative,
            progress,
            speed=float(environment.reference_speed),
            orientation=np.asarray(environment.gates[gate_index].orientation, dtype=np.float64),
            dt=dt,
            horizon=controller.horizon,
        )
        state = np.asarray(arrays["state_X"][index], dtype=np.float64)
        expected_input = build_policy_input(
            state,
            window,
            input_dim=len(PATH_AWARE_INPUT_FEATURES),
            reference_speed=float(environment.reference_speed),
        )
        expected_control = controller.solve(state, window)
        max_input_error = max(max_input_error, float(np.max(np.abs(expected_input - arrays["X"][index]))))
        max_label_error = max(max_label_error, float(np.max(np.abs(expected_control - arrays["U"][index]))))
    families, family_counts = np.unique(arrays["sample_family"], return_counts=True)
    gates, gate_counts = np.unique(arrays["target_gate_id"], return_counts=True)
    unique_x = np.unique(arrays["X"], axis=0).shape[0]
    result = {
        "status": "valid",
        "root": str(root),
        "final_dataset": str(final_path),
        "samples": rows,
        "X_shape": list(arrays["X"].shape),
        "U_shape": list(arrays["U"].shape),
        "environments": int(np.unique(arrays["environment_id"]).size),
        "unique_input_rows": int(unique_x),
        "duplicate_input_fraction": float(1.0 - unique_x / max(rows, 1)),
        "sample_family_counts": {str(int(k)): int(v) for k, v in zip(families, family_counts)},
        "target_gate_counts": {str(int(k)): int(v) for k, v in zip(gates, gate_counts)},
        "label_checks": check_count,
        "max_recomputed_input_abs_error": max_input_error,
        "max_recomputed_label_abs_error": max_label_error,
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
