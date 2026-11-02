"""Atomic episode storage, resume manifests, merging and validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


CORE_KEYS = ("X", "U", "episode_id", "initial_state_id", "environment_id", "step_id")
METADATA_KEYS = (
    "target_gate_id",
    "timestamp",
    "position",
    "quaternion",
    "velocity",
    "acceleration",
    "reference_position",
    "reference_orientation",
    "mpc_solve_time",
    "actual_loop_period",
    "deadline_miss",
)
OPTIONAL_METADATA_KEYS = (
    # Four future route samples, each represented by a position relative to
    # the measured drone position.  This is deliberately separate from X so
    # the original 12-D baseline remains directly reusable.
    "path_context",
    "reference_speed",
    # 0=nominal MPC rollout, 1=state-perturbation rollout, 2=DAgger rollout.
    "data_source",
    "perturbation",
    "student_control",
    "shield_active",
)


@dataclass
class EpisodeSamples:
    """One fixed-length teacher episode and its aligned metadata."""

    X: np.ndarray
    U: np.ndarray
    episode_id: int
    initial_state_id: int
    environment_id: int
    step_id: np.ndarray
    target_gate_id: np.ndarray
    timestamp: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    reference_position: np.ndarray
    reference_orientation: np.ndarray
    mpc_solve_time: np.ndarray
    actual_loop_period: np.ndarray
    deadline_miss: np.ndarray
    path_context: np.ndarray | None = None
    reference_speed: np.ndarray | None = None
    data_source: np.ndarray | None = None
    perturbation: np.ndarray | None = None
    student_control: np.ndarray | None = None
    shield_active: np.ndarray | None = None

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "X": np.asarray(self.X, dtype=np.float64),
            "U": np.asarray(self.U, dtype=np.float64),
            "episode_id": np.full(len(self.X), self.episode_id, dtype=np.int64),
            "initial_state_id": np.full(len(self.X), self.initial_state_id, dtype=np.int64),
            "environment_id": np.full(len(self.X), self.environment_id, dtype=np.int64),
            "step_id": np.asarray(self.step_id, dtype=np.int64),
            "target_gate_id": np.asarray(self.target_gate_id, dtype=np.int64),
            "timestamp": np.asarray(self.timestamp, dtype=np.float64),
            "position": np.asarray(self.position, dtype=np.float64),
            "quaternion": np.asarray(self.quaternion, dtype=np.float64),
            "velocity": np.asarray(self.velocity, dtype=np.float64),
            "acceleration": np.asarray(self.acceleration, dtype=np.float64),
            "reference_position": np.asarray(self.reference_position, dtype=np.float64),
            "reference_orientation": np.asarray(self.reference_orientation, dtype=np.float64),
            "mpc_solve_time": np.asarray(self.mpc_solve_time, dtype=np.float64),
            "actual_loop_period": np.asarray(self.actual_loop_period, dtype=np.float64),
            "deadline_miss": np.asarray(self.deadline_miss, dtype=np.int8),
        }
        if self.path_context is not None:
            arrays["path_context"] = np.asarray(self.path_context, dtype=np.float64)
        if self.reference_speed is not None:
            arrays["reference_speed"] = np.asarray(self.reference_speed, dtype=np.float64)
        if self.data_source is not None:
            arrays["data_source"] = np.asarray(self.data_source, dtype=np.int8)
        if self.perturbation is not None:
            arrays["perturbation"] = np.asarray(self.perturbation, dtype=np.float64)
        if self.student_control is not None:
            arrays["student_control"] = np.asarray(self.student_control, dtype=np.float64)
        if self.shield_active is not None:
            arrays["shield_active"] = np.asarray(self.shield_active, dtype=np.int8)
        return arrays


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        writer(temporary)
        last_error: OSError | None = None
        for _ in range(5):
            try:
                os.replace(temporary, path)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        if last_error is not None:
            raise last_error
    finally:
        if temporary.exists():
            temporary.unlink()


def save_episode_atomic(path: str | Path, samples: EpisodeSamples) -> Path:
    output = Path(path)
    arrays = samples.arrays()
    validate_episode_arrays(arrays, expected_steps=len(samples.X))

    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    _atomic_replace(output, write)
    return output


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_episode_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_steps: int,
    expected_episode_id: int | None = None,
    check_step_sequence: bool = True,
) -> None:
    missing = [key for key in CORE_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"episode is missing core arrays: {missing}")
    sample_count = int(np.asarray(arrays["X"]).shape[0])
    if sample_count != expected_steps:
        raise ValueError(f"episode has {sample_count} samples; expected {expected_steps}")
    x_shape = np.asarray(arrays["X"]).shape
    if len(x_shape) != 2 or x_shape[0] != expected_steps or x_shape[1] < 1:
        raise ValueError("episode X must have shape (steps, input_dimensions)")
    if np.asarray(arrays["U"]).shape != (expected_steps, 4):
        raise ValueError("episode U must have shape (steps, 4)")
    for key in CORE_KEYS + METADATA_KEYS + OPTIONAL_METADATA_KEYS:
        if key in arrays and np.asarray(arrays[key]).shape[0] != expected_steps:
            raise ValueError(f"episode array {key} is not aligned to {expected_steps} samples")
    if not np.isfinite(np.asarray(arrays["X"], dtype=np.float64)).all():
        raise ValueError("episode X contains NaN or Inf")
    if not np.isfinite(np.asarray(arrays["U"], dtype=np.float64)).all():
        raise ValueError("episode U contains NaN or Inf")
    for key in ("path_context", "reference_speed"):
        if key in arrays and not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise ValueError(f"episode {key} contains NaN or Inf")
    if "path_context" in arrays:
        path_context = np.asarray(arrays["path_context"])
        if path_context.ndim != 3 or path_context.shape[2] != 3:
            raise ValueError("episode path_context must have shape (steps, context_points, 3)")
    if "perturbation" in arrays:
        perturbation = np.asarray(arrays["perturbation"])
        if perturbation.ndim != 2 or perturbation.shape[1] != 3:
            raise ValueError("episode perturbation must have shape (steps, 3)")
        if not np.isfinite(perturbation.astype(np.float64)).all():
            raise ValueError("episode perturbation contains NaN or Inf")
    if "student_control" in arrays:
        student_control = np.asarray(arrays["student_control"])
        if student_control.ndim != 2 or student_control.shape[1] != 4:
            raise ValueError("episode student_control must have shape (steps, 4)")
        if not np.isfinite(student_control.astype(np.float64)).all():
            raise ValueError("episode student_control contains NaN or Inf")
    if "shield_active" in arrays:
        shield_active = np.asarray(arrays["shield_active"])
        if shield_active.ndim != 1:
            raise ValueError("episode shield_active must have shape (steps,)")
    if check_step_sequence:
        np.testing.assert_array_equal(np.asarray(arrays["step_id"]), np.arange(expected_steps))
    if expected_episode_id is not None:
        np.testing.assert_array_equal(
            np.asarray(arrays["episode_id"]),
            np.full(expected_steps, expected_episode_id),
        )


def is_valid_episode(
    path: str | Path,
    *,
    expected_steps: int,
    expected_episode_id: int | None = None,
) -> bool:
    try:
        validate_episode_arrays(
            load_episode(path),
            expected_steps=expected_steps,
            expected_episode_id=expected_episode_id,
        )
    except (OSError, ValueError, KeyError, AssertionError, EOFError, zipfile.BadZipFile):
        return False
    return True


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    _atomic_replace(
        output,
        lambda temporary: temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        ),
    )
    return output


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_episodes(
    episode_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    expected_steps: int,
    expected_episode_count: int,
    metadata_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(episode_paths) != expected_episode_count:
        raise ValueError(f"expected {expected_episode_count} episode paths, got {len(episode_paths)}")
    arrays_by_episode = []
    for episode_id, path in enumerate(episode_paths):
        arrays = load_episode(path)
        validate_episode_arrays(
            arrays,
            expected_steps=expected_steps,
            expected_episode_id=episode_id,
        )
        arrays_by_episode.append(arrays)
    merged: dict[str, np.ndarray] = {}
    keys = list(CORE_KEYS + METADATA_KEYS + OPTIONAL_METADATA_KEYS)
    for key in keys:
        if key not in arrays_by_episode[0]:
            continue
        merged[key] = np.concatenate([arrays[key] for arrays in arrays_by_episode], axis=0)
    validate_episode_arrays(
        merged,
        expected_steps=expected_steps * expected_episode_count,
        check_step_sequence=False,
    )
    output = Path(output_path)

    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **merged)

    _atomic_replace(output, write)
    if metadata_csv_path is not None:
        write_metadata_csv(metadata_csv_path, merged)
    return dataset_statistics(merged)


def write_metadata_csv(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    output = Path(path)
    rows = int(np.asarray(arrays["X"]).shape[0])
    fields = [
        "episode_id",
        "initial_state_id",
        "environment_id",
        "step_id",
        "target_gate_id",
        "timestamp",
        "mpc_solve_time",
        "actual_loop_period",
        "deadline_miss",
    ]
    if "reference_speed" in arrays:
        fields.append("reference_speed")
    if "data_source" in arrays:
        fields.append("data_source")
    _atomic_replace(
        output,
        lambda temporary: _write_metadata_csv_file(temporary, fields, arrays, rows),
    )
    return output


def _write_metadata_csv_file(
    path: Path,
    fields: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    rows: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for index in range(rows):
            writer.writerow({field: np.asarray(arrays[field])[index].item() for field in fields})


def dataset_statistics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    x = np.asarray(arrays["X"], dtype=np.float64)
    u = np.asarray(arrays["U"], dtype=np.float64)
    x_min_per_dim = np.min(x, axis=0)
    x_max_per_dim = np.max(x, axis=0)
    u_min_per_dim = np.min(u, axis=0)
    u_max_per_dim = np.max(u, axis=0)
    stats: dict[str, Any] = {
        "samples": int(x.shape[0]),
        "X_shape": list(x.shape),
        "U_shape": list(u.shape),
        "X_min": float(np.min(x)),
        "X_max": float(np.max(x)),
        "X_mean": float(np.mean(x)),
        "X_std": float(np.std(x)),
        "X_min_per_dim": x_min_per_dim.tolist(),
        "X_max_per_dim": x_max_per_dim.tolist(),
        "U_min": float(np.min(u)),
        "U_max": float(np.max(u)),
        "U_mean": float(np.mean(u)),
        "U_std": float(np.std(u)),
        "U_min_per_dim": u_min_per_dim.tolist(),
        "U_max_per_dim": u_max_per_dim.tolist(),
        "zero_input_dimensions": [
            int(index) for index in range(x.shape[1]) if np.allclose(x[:, index], 0.0)
        ],
        "constant_input_dimensions": [
            int(index) for index in range(x.shape[1]) if np.ptp(x[:, index]) <= 1e-12
        ],
        "controls_identical": bool(np.allclose(u, u[0:1, :])),
        "deadline_misses": int(np.sum(np.asarray(arrays.get("deadline_miss", [])))),
        "average_mpc_solve_time": float(
            np.mean(np.asarray(arrays["mpc_solve_time"], dtype=np.float64))
        )
        if "mpc_solve_time" in arrays
        else 0.0,
        "unique_initial_states": int(np.unique(arrays["initial_state_id"]).size),
        "unique_environments": int(np.unique(arrays["environment_id"]).size),
        "unique_episodes": int(np.unique(arrays["episode_id"]).size),
    }
    if "reference_speed" in arrays:
        stats["reference_speed_values"] = sorted(
            float(value) for value in np.unique(np.asarray(arrays["reference_speed"], dtype=np.float64))
        )
    if "data_source" in arrays:
        values, counts = np.unique(np.asarray(arrays["data_source"], dtype=np.int8), return_counts=True)
        stats["data_source_counts"] = {
            str(int(value)): int(count) for value, count in zip(values, counts)
        }
    if "path_context" in arrays:
        stats["path_context_shape"] = list(np.asarray(arrays["path_context"]).shape)
    return stats


def validate_merged_dataset(
    path: str | Path,
    *,
    expected_samples: int,
    expected_episodes: int,
    expected_steps: int,
    expected_initial_states: int,
    expected_environments: int,
    expected_input_dimensions: int | None = None,
) -> dict[str, Any]:
    arrays = load_episode(path)
    validate_episode_arrays(
        arrays,
        expected_steps=expected_samples,
        check_step_sequence=False,
    )
    if arrays["X"].ndim != 2 or arrays["X"].shape[0] != expected_samples:
        raise ValueError("merged X shape is incorrect")
    if (
        expected_input_dimensions is not None
        and arrays["X"].shape[1] != expected_input_dimensions
    ):
        raise ValueError("merged X input dimension is incorrect")
    if arrays["U"].shape != (expected_samples, 4):
        raise ValueError("merged U shape is incorrect")
    if np.unique(arrays["episode_id"]).size != expected_episodes:
        raise ValueError("merged episode count is incorrect")
    if np.unique(arrays["initial_state_id"]).size != expected_initial_states:
        raise ValueError("merged initial-state count is incorrect")
    if np.unique(arrays["environment_id"]).size != expected_environments:
        raise ValueError("merged environment count is incorrect")
    for episode_id in range(expected_episodes):
        episode_rows = np.flatnonzero(arrays["episode_id"] == episode_id)
        if episode_rows.size != expected_steps:
            raise ValueError(f"episode {episode_id} does not have {expected_steps} samples")
        np.testing.assert_array_equal(arrays["step_id"][episode_rows], np.arange(expected_steps))
    return dataset_statistics(arrays)


def write_summary(path: str | Path, summary: Mapping[str, Any]) -> Path:
    output = Path(path)
    lines = ["Teacher dataset collection summary", ""]
    for key, value in summary.items():
        label = str(key).replace("_", " ").title()
        lines.append(f"{label}: {value}")
    _atomic_replace(output, lambda temporary: temporary.write_text("\n".join(lines) + "\n", encoding="utf-8"))
    return output
