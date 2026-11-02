"""Independent recovery-state queries labelled by the paper-fidelity MPC.

This is a robustness extension, not part of FALCON's published 40k protocol.
Each row is sampled independently, the future polynomial is regenerated from
that row, and the MPC is solved again.  No perturbed input reuses an old label.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from airsim_interface import RawDroneState
from dataset_store import atomic_write_json

from .contracts import INPUT_CONTRACT, INPUT_FEATURES, OUTPUT_FEATURES, build_gate_input
from .acado_mpc import AcadoMPCConfig, AcadoRtiMPC
from .motion_primitive import (
    MotionPrimitiveConfig,
    MotionPrimitiveInfeasible,
    QuinticMotionPrimitivePlanner,
)
from .nonlinear_mpc import CasadiTrackingMPC, NonlinearMPCConfig


FAMILY_NAMES = (
    "nominal_tracking",
    "lateral_recovery",
    "vertical_recovery",
    "attitude_recovery",
    "high_speed_recovery",
    "combined_boundary_recovery",
)
CONTROL_BUCKET_NAMES = ("interior", "near_limit", "saturated")


def _hash(config: Mapping[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rotation_vector_to_quaternion(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        quaternion = np.array([1.0, *(0.5 * vector)], dtype=np.float64)
    else:
        quaternion = np.array(
            [np.cos(angle / 2.0), *(vector * (np.sin(angle / 2.0) / angle))],
            dtype=np.float64,
        )
    return quaternion / np.linalg.norm(quaternion)


def _signed_edge(rng: np.random.Generator, low: float, high: float) -> float:
    magnitude = rng.uniform(low, high)
    return float(magnitude if rng.random() < 0.5 else -magnitude)


def sample_recovery_input(rng: np.random.Generator, family: int) -> np.ndarray:
    """Sample one finite 12-D state from an explicit recovery family."""

    if family < 0 or family >= len(FAMILY_NAMES):
        raise ValueError("invalid recovery family")
    position_error = np.array(
        [rng.uniform(0.2, 16.0), rng.normal(0.0, 0.25), rng.normal(0.0, 0.2)],
        dtype=np.float64,
    )
    attitude_error = rng.normal(0.0, 0.06, size=3)
    velocity = np.array(
        [rng.uniform(0.0, 6.0), rng.normal(0.0, 0.35), rng.normal(0.0, 0.3)],
        dtype=np.float64,
    )
    acceleration = rng.normal(0.0, 0.6, size=3)
    if family == 1:
        position_error[1] = _signed_edge(rng, 0.5, 3.0)
        velocity[1] = _signed_edge(rng, 0.3, 2.5)
    elif family == 2:
        position_error[2] = _signed_edge(rng, 0.4, 2.4)
        velocity[2] = _signed_edge(rng, 0.3, 2.0)
    elif family == 3:
        attitude_error = rng.uniform(-0.45, 0.45, size=3)
    elif family == 4:
        velocity[0] = rng.uniform(-2.5, 8.0)
        velocity[1:3] = rng.uniform(-2.5, 2.5, size=2)
        acceleration = rng.uniform(-3.5, 3.5, size=3)
    elif family == 5:
        position_error[0] = rng.choice((rng.uniform(0.05, 1.0), rng.uniform(12.0, 16.0)))
        position_error[1] = _signed_edge(rng, 1.0, 3.0)
        position_error[2] = _signed_edge(rng, 0.8, 2.4)
        attitude_error = rng.uniform(-0.4, 0.4, size=3)
        velocity = rng.uniform([-2.5, -2.5, -2.0], [8.0, 2.5, 2.0])
        acceleration = rng.uniform(-4.0, 4.0, size=3)
    value = np.concatenate((position_error, attitude_error, velocity, acceleration))
    if value.shape != (12,) or not np.isfinite(value).all():
        raise RuntimeError("recovery sampler produced an invalid row")
    return value


def sample_capacity_balanced_input(
    rng: np.random.Generator,
    family: int,
    target_bucket: int,
) -> np.ndarray:
    """Propose a state for a requested output-control bucket."""

    if target_bucket == 2:
        return sample_recovery_input(rng, family)
    near = target_bucket == 1
    position_error = np.array(
        [
            rng.uniform(0.1, 16.0),
            rng.normal(0.0, 0.4 if near else 0.15),
            rng.normal(0.0, 0.3 if near else 0.12),
        ],
        dtype=np.float64,
    )
    attitude_error = rng.normal(0.0, 0.10 if near else 0.04, size=3)
    velocity = np.array(
        [
            rng.uniform(0.4, 1.9) if near else rng.uniform(0.0, 1.2),
            rng.normal(0.0, 0.7 if near else 0.2),
            rng.normal(0.0, 0.6 if near else 0.15),
        ],
        dtype=np.float64,
    )
    acceleration = rng.normal(0.0, 1.2 if near else 0.4, size=3)
    scale = 1.0 if near else 0.55
    if family == 1:
        position_error[1] = _signed_edge(rng, 0.35 * scale, 2.0 * scale)
        velocity[1] = _signed_edge(rng, 0.15 * scale, 1.4 * scale)
    elif family == 2:
        position_error[2] = _signed_edge(rng, 0.3 * scale, 1.6 * scale)
        velocity[2] = _signed_edge(rng, 0.15 * scale, 1.2 * scale)
    elif family == 3:
        limit = 0.30 if near else 0.15
        attitude_error = rng.uniform(-limit, limit, size=3)
    elif family == 4:
        velocity[0] = rng.uniform(-1.5, 1.5) if near else rng.uniform(-1.0, 0.8)
        cross_limit = 1.6 if near else 0.7
        velocity[1:3] = rng.uniform(-cross_limit, cross_limit, size=2)
        limit = 2.5 if near else 1.0
        acceleration = rng.uniform(-limit, limit, size=3)
    elif family == 5:
        position_error[0] = rng.uniform(0.1, 16.0)
        position_error[1] = _signed_edge(rng, 0.4 * scale, 2.0 * scale)
        position_error[2] = _signed_edge(rng, 0.3 * scale, 1.5 * scale)
        attitude_limit = 0.28 if near else 0.13
        attitude_error = rng.uniform(-attitude_limit, attitude_limit, size=3)
        velocity[0] = rng.uniform(-2.0, 2.0) if near else rng.uniform(-1.5, 1.5)
        cross_limit = 1.5 if near else 0.6
        velocity[1:3] = rng.uniform(-cross_limit, cross_limit, size=2)
        acceleration_limit = 2.5 if near else 1.0
        acceleration = rng.uniform(-acceleration_limit, acceleration_limit, size=3)
    value = np.concatenate((position_error, attitude_error, velocity, acceleration))
    if value.shape != (12,) or not np.isfinite(value).all():
        raise RuntimeError("capacity-balanced sampler produced an invalid row")
    return value


def classify_control_bucket(
    control: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    saturation_tolerance: float,
    near_limit_fraction: float,
) -> int:
    value = np.asarray(control, dtype=np.float64).reshape(4)
    if np.any(value <= lower + saturation_tolerance) or np.any(
        value >= upper - saturation_tolerance
    ):
        return 2
    normalized_distance = np.minimum(
        (value - lower) / (upper - lower),
        (upper - value) / (upper - lower),
    )
    return 1 if float(np.min(normalized_distance)) <= near_limit_fraction else 0


def raw_state_from_input(x: np.ndarray, gate_position: np.ndarray) -> RawDroneState:
    value = np.asarray(x, dtype=np.float64).reshape(12)
    relative = _rotation_vector_to_quaternion(value[3:6])
    # q_rel = q_gate * inverse(q_current), and q_gate is identity.
    current_q = np.array([relative[0], -relative[1], -relative[2], -relative[3]])
    return RawDroneState(
        position=np.asarray(gate_position, dtype=np.float64) - value[0:3],
        quaternion=current_q,
        velocity=value[6:9],
        acceleration=value[9:12],
        angular_velocity=np.zeros(3),
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _make_controller(config: Mapping[str, Any], config_file: Path) -> tuple[Any, str]:
    values = dict(config["mpc"])
    backend = str(values.pop("backend", "casadi_nonlinear_mpc")).lower()
    if backend in {
        "official_rpg_mpc_acado_qpoases",
        "acado",
        "acado_qpoases",
    }:
        library = Path(str(values.get("library", ""))).expanduser()
        if not library.is_absolute():
            library = (config_file.parent / library).resolve()
        values["library"] = str(library)
        return AcadoRtiMPC(AcadoMPCConfig(**values)), "official_rpg_mpc_acado_qpoases"
    if backend not in {"casadi", "casadi_nonlinear_mpc"}:
        raise ValueError(f"unsupported MPC backend: {backend}")
    values["dt"] = float(config["timing"]["dt"])
    return CasadiTrackingMPC(NonlinearMPCConfig(**values)), "casadi_nonlinear_mpc"


def generate(config_path: str | Path, *, resume: bool = False) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    output = (config_file.parent / config["dataset"]["root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    signature = _hash(config)
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != signature:
            raise ValueError("resume config differs from the recorded recovery config")
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "strategy": "falcon_acado_12d_independent_state_queries_v2",
            "paper_core": False,
            "input_contract": INPUT_CONTRACT,
            "input_dimensions": 12,
            "input_features": list(INPUT_FEATURES),
            "output_dimensions": 4,
            "output_features": list(OUTPUT_FEATURES),
            "config_sha256": signature,
            "sample_count": int(config["dataset"]["sample_count"]),
            "chunk_size": int(config["dataset"]["chunk_size"]),
            "completed_chunks": [],
            "families": list(FAMILY_NAMES),
            "warning": "robustness extension; do not report as the paper's 40k dataset",
        }
        atomic_write_json(manifest_path, manifest)
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    controller, backend = _make_controller(config, config_file)
    manifest["mpc_backend"] = backend
    manifest["planner_signature"] = planner.signature()
    count = int(config["dataset"]["sample_count"])
    chunk_size = int(config["dataset"]["chunk_size"])
    seed = int(config["random_seed"])
    horizon = int(config["mpc"]["horizon"])
    dt = float(config["timing"]["dt"])
    query_group_size = int(config["dataset"].get("query_group_size", 100))
    max_sample_attempts = int(config["dataset"].get("max_sample_attempts", 100))
    if query_group_size < 1 or max_sample_attempts < 1:
        raise ValueError("query_group_size and max_sample_attempts must be positive")
    gate_position = np.array([0.0, 0.0, -3.0], dtype=np.float64)
    gate_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sampling = dict(config.get("sampling", {}))
    sampling_profile = str(sampling.get("profile", "aggressive_v1"))
    bucket_pattern: list[int] = []
    if sampling_profile == "capacity_balanced_v1":
        counts = sampling.get(
            "bucket_slots", {"interior": 12, "near_limit": 5, "saturated": 3}
        )
        bucket_pattern = [
            bucket
            for bucket, name in enumerate(CONTROL_BUCKET_NAMES)
            for _ in range(int(counts[name]))
        ]
        if not bucket_pattern:
            raise ValueError("capacity-balanced sampling needs non-empty bucket slots")
    elif sampling_profile != "aggressive_v1":
        raise ValueError(f"unsupported sampling profile: {sampling_profile}")
    lower = np.array(
        [
            float(config["mpc"]["min_thrust"]),
            -float(config["mpc"]["max_body_rate_xy"]),
            -float(config["mpc"]["max_body_rate_xy"]),
            -float(config["mpc"]["max_body_rate_z"]),
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            float(config["mpc"]["max_thrust"]),
            float(config["mpc"]["max_body_rate_xy"]),
            float(config["mpc"]["max_body_rate_xy"]),
            float(config["mpc"]["max_body_rate_z"]),
        ],
        dtype=np.float64,
    )
    saturation_tolerance = float(sampling.get("saturation_tolerance", 1e-3))
    near_limit_fraction = float(sampling.get("near_limit_fraction", 0.15))
    manifest["sampling_profile"] = sampling_profile
    manifest["control_bucket_names"] = list(CONTROL_BUCKET_NAMES)
    total_chunks = (count + chunk_size - 1) // chunk_size
    for chunk_id in range(total_chunks):
        path = output / "chunks" / f"chunk_{chunk_id:05d}.npz"
        start = chunk_id * chunk_size
        stop = min(count, start + chunk_size)
        expected = stop - start
        if chunk_id in manifest.get("completed_chunks", []) and path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if archive["X"].shape == (expected, 12) and archive["U"].shape == (expected, 4):
                    continue
        rng = np.random.default_rng(seed + 104729 * chunk_id)
        x_rows: list[np.ndarray] = []
        u_rows: list[np.ndarray] = []
        families: list[int] = []
        control_buckets: list[int] = []
        solve_times: list[float] = []
        rejected_before_accept: list[int] = []
        rejected_samples = 0
        for local_index in range(expected):
            global_index = start + local_index
            family = global_index % len(FAMILY_NAMES)
            if bucket_pattern:
                family_cycle = global_index // len(FAMILY_NAMES)
                target_bucket = bucket_pattern[
                    (family_cycle + 7 * family) % len(bucket_pattern)
                ]
            else:
                target_bucket = -1
            last_error: Exception | None = None
            attempts_for_row = 0
            for _ in range(max_sample_attempts):
                x = (
                    sample_capacity_balanced_input(rng, family, target_bucket)
                    if target_bucket >= 0
                    else sample_recovery_input(rng, family)
                )
                raw = raw_state_from_input(x, gate_position)
                np.testing.assert_allclose(
                    build_gate_input(raw, gate_position, gate_orientation),
                    x,
                    rtol=1e-10,
                    atol=1e-10,
                )
                try:
                    references = planner.reference_window(
                        position=raw.position,
                        velocity=raw.velocity,
                        acceleration=raw.acceleration,
                        gate_position=gate_position,
                        gate_orientation=gate_orientation,
                        dt=dt,
                        horizon=horizon,
                    )
                    controller.reset()
                    started = time.perf_counter()
                    control = controller.solve(raw, references)
                    solve_time = time.perf_counter() - started
                except (MotionPrimitiveInfeasible, RuntimeError) as exc:
                    last_error = exc
                    rejected_samples += 1
                    attempts_for_row += 1
                    continue
                if not np.isfinite(control).all():
                    last_error = RuntimeError("MPC returned non-finite control")
                    rejected_samples += 1
                    attempts_for_row += 1
                    continue
                actual_bucket = classify_control_bucket(
                    control,
                    lower,
                    upper,
                    saturation_tolerance=saturation_tolerance,
                    near_limit_fraction=near_limit_fraction,
                )
                if target_bucket >= 0 and actual_bucket != target_bucket:
                    last_error = RuntimeError(
                        f"control bucket {CONTROL_BUCKET_NAMES[actual_bucket]} did not match "
                        f"target {CONTROL_BUCKET_NAMES[target_bucket]}"
                    )
                    rejected_samples += 1
                    attempts_for_row += 1
                    continue
                break
            else:
                raise RuntimeError(
                    f"could not generate sample {global_index} family={FAMILY_NAMES[family]} "
                    f"after {max_sample_attempts} attempts: {last_error}"
                )
            solve_times.append(solve_time)
            rejected_before_accept.append(attempts_for_row)
            x_rows.append(x)
            u_rows.append(control)
            families.append(family)
            control_buckets.append(actual_bucket)
        _atomic_npz(
            path,
            {
                "X": np.asarray(x_rows, dtype=np.float32),
                "U": np.asarray(u_rows, dtype=np.float32),
                "sample_id": np.arange(start, stop, dtype=np.int64),
                "query_group_id": (
                    np.arange(start, stop, dtype=np.int64) // query_group_size
                ),
                "sample_family": np.asarray(families, dtype=np.int8),
                "control_bucket": np.asarray(control_buckets, dtype=np.int8),
                "data_source": np.ones(expected, dtype=np.int8),
                "mpc_solve_time": np.asarray(solve_times, dtype=np.float64),
                "reference_speed": np.full(expected, float(config["planner"]["nominal_speed"]), dtype=np.float32),
                "rejected_attempts": np.asarray(rejected_before_accept, dtype=np.int16),
            },
        )
        completed = set(int(item) for item in manifest.get("completed_chunks", []))
        completed.add(chunk_id)
        manifest["completed_chunks"] = sorted(completed)
        manifest["completed_samples"] = min(count, len(completed) * chunk_size)
        manifest["query_group_size"] = query_group_size
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        print(f"Recovery chunks: {len(completed)}/{total_chunks} | samples through {stop}/{count}", flush=True)
    arrays: dict[str, list[np.ndarray]] = {}
    for chunk_id in range(total_chunks):
        with np.load(output / "chunks" / f"chunk_{chunk_id:05d}.npz", allow_pickle=False) as archive:
            for key in archive.files:
                arrays.setdefault(key, []).append(np.asarray(archive[key]))
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    final = output / "final" / str(config["dataset"]["output_name"])
    _atomic_npz(final, merged)
    manifest.update(
        status="completed",
        completed_samples=count,
        final_dataset=str(Path("final") / final.name),
        X_shape=list(merged["X"].shape),
        U_shape=list(merged["U"].shape),
        family_counts={
            FAMILY_NAMES[index]: int(np.count_nonzero(merged["sample_family"] == index))
            for index in range(len(FAMILY_NAMES))
        },
        control_bucket_counts={
            CONTROL_BUCKET_NAMES[index]: int(
                np.count_nonzero(merged["control_bucket"] == index)
            )
            for index in range(len(CONTROL_BUCKET_NAMES))
        },
        mean_mpc_solve_time_seconds=float(np.mean(merged["mpc_solve_time"])),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(manifest_path, manifest)
    print(f"Recovery extension complete: X={merged['X'].shape} U={merged['U'].shape}")
    return output
