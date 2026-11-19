"""Generate the 120k phase-consistent ACADO augmentation for hybrid_160k_v1.

The immutable 40k AirSim core supplies real states, environments and globally
continuous reference trajectories.  This generator never changes that core.
It draws perturbations around a core trajectory *phase*, recomputes the ACADO
teacher label, and writes three balanced extensions:

* 40k independent recovery queries;
* 40k five-step local rollouts following an immutable global reference phase;
* 40k near-gate / gate-transition queries.

All rows use ``[p_ref-p, rotvec(q_ref q^-1), v_ref-v, a_ref]``.  In
particular, no gate-relative and trajectory-relative vectors are mixed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState
from dataset_store import atomic_write_json
from falcon_paper_fidelity.contracts import (
    OUTPUT_FEATURES,
    TRAJECTORY_INPUT_CONTRACT,
    TRAJECTORY_INPUT_FEATURES,
    build_trajectory_input,
)
from falcon_paper_fidelity.motion_primitive import (
    MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner,
)
from falcon_paper_fidelity.phase_recovery_dataset import (
    _build_trajectories,
    _controller,
    _load_gates,
    _perturb_state,
    _validate_reconstruction,
)
from falcon_paper_fidelity.recovery_dataset import (
    CONTROL_BUCKET_NAMES,
    classify_control_bucket,
)
from falcon_paper_fidelity.short_rollout_dataset import propagate_state


SOURCE_RECOVERY = 1
SOURCE_SHORT_ROLLOUT = 2
SOURCE_CRITICAL_PHASE = 3


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _hash(value: Mapping[str, Any]) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _finite_state(state: RawDroneState, *, max_speed: float, max_acceleration: float) -> bool:
    values = (state.position, state.quaternion, state.velocity, state.acceleration)
    return (
        all(np.isfinite(value).all() for value in values)
        and float(np.linalg.norm(state.velocity)) <= max_speed + 1e-8
        and float(np.linalg.norm(state.acceleration)) <= max_acceleration + 1e-8
    )


def _state_with_launch_speed(state: RawDroneState, rng: np.random.Generator) -> RawDroneState:
    """Keep the core trajectory phase but explicitly cover real takeoff speeds."""

    velocity = np.array(
        [rng.uniform(-0.4, 3.5), rng.uniform(-0.35, 0.35), rng.uniform(-0.30, 0.30)],
        dtype=np.float64,
    )
    acceleration = np.array(
        [rng.uniform(-1.2, 1.2), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
        dtype=np.float64,
    )
    return RawDroneState(
        position=state.position.copy(), quaternion=state.quaternion.copy(),
        velocity=velocity, acceleration=acceleration,
        angular_velocity=state.angular_velocity.copy(),
    )


def _append(rows: dict[str, list[Any]], *, x: np.ndarray, u: np.ndarray, state: RawDroneState,
            source: Mapping[str, np.ndarray], base: int, group_id: int, rollout_step: int,
            source_kind: int, family: int, bucket: int, solve_seconds: float,
            reference: Any) -> None:
    rows["X"].append(x)
    rows["U"].append(u)
    rows["sample_id"].append(len(rows["sample_id"]))
    rows["query_group_id"].append(group_id)
    rows["rollout_step"].append(rollout_step)
    rows["source_kind"].append(source_kind)
    rows["sample_family"].append(family)
    rows["control_bucket"].append(bucket)
    rows["episode_id"].append(int(source["episode_id"][base]))
    rows["initial_state_id"].append(int(source["initial_state_id"][base]))
    rows["environment_id"].append(int(source["environment_id"][base]))
    rows["step_id"].append(int(source["step_id"][base]) + rollout_step)
    rows["target_gate_id"].append(int(source["target_gate_id"][base]))
    rows["position"].append(state.position.copy())
    rows["quaternion"].append(state.quaternion.copy())
    rows["velocity"].append(state.velocity.copy())
    rows["acceleration"].append(state.acceleration.copy())
    rows["reference_position"].append(np.asarray(reference.position, dtype=np.float64).copy())
    rows["reference_orientation"].append(np.asarray(reference.orientation, dtype=np.float64).copy())
    rows["reference_speed"].append(10.0)
    rows["mpc_solve_time"].append(solve_seconds)
    rows["data_source"].append(source_kind)


def _bucket_pattern(slots: Mapping[str, Any], *, allowed: tuple[int, ...]) -> tuple[int, ...]:
    """Expand a named control-bucket recipe into a deterministic cycle.

    Sampling a state first and only rejecting it once the global saturation
    quota is exhausted is biased and can deadlock.  Instead, each group gets a
    requested difficulty bucket before its state is drawn.  The sampler then
    retries locally until the MPC label actually belongs to that bucket.
    """

    pattern: list[int] = []
    for bucket, name in enumerate(CONTROL_BUCKET_NAMES):
        count = int(slots.get(name, 0))
        if count < 0:
            raise ValueError(f"control_bucket_slots.{name} must be non-negative")
        if count and bucket not in allowed:
            raise ValueError(f"control_bucket_slots requests unsupported bucket {name!r}")
        pattern.extend([bucket] * count)
    if not pattern:
        raise ValueError("control bucket slot recipe must contain at least one slot")
    return tuple(pattern)


def _build_pools(source: Mapping[str, np.ndarray], environments: Mapping[int, Any]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return all, low-speed and near/transition base-row pools per environment."""

    all_rows: dict[int, list[int]] = {int(key): [] for key in environments}
    low_rows: dict[int, list[int]] = {int(key): [] for key in environments}
    critical_rows: dict[int, list[int]] = {int(key): [] for key in environments}
    last_gate_by_episode: dict[int, int] = {}
    for index in range(len(source["X"])):
        env = int(source["environment_id"][index])
        episode = int(source["episode_id"][index])
        all_rows[env].append(index)
        speed = float(np.linalg.norm(source["velocity"][index]))
        if speed <= 4.0 or int(source["step_id"][index]) < 12:
            low_rows[env].append(index)
        gate_id = int(source["target_gate_id"][index])
        gate = environments[env][gate_id - 1]
        distance = float(np.linalg.norm(source["position"][index] - gate.center))
        changed = episode in last_gate_by_episode and last_gate_by_episode[episode] != gate_id
        if distance <= 5.0 or changed:
            critical_rows[env].append(index)
        last_gate_by_episode[episode] = gate_id
    return (
        {key: np.asarray(value, dtype=np.int64) for key, value in all_rows.items()},
        {key: np.asarray(value if value else all_rows[key], dtype=np.int64) for key, value in low_rows.items()},
        {key: np.asarray(value if value else all_rows[key], dtype=np.int64) for key, value in critical_rows.items()},
    )


def generate(config_path: str | Path, *, resume: bool = False, smoke_test: bool = False) -> Path:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    base = config_file.parent
    if smoke_test:
        config = json.loads(json.dumps(config))
        config["dataset"].update(
            root=str(config["dataset"]["root"]) + "_smoke",
            output_name="falcon_hybrid_extensions_smoke.npz",
            recovery_queries=1200,
            short_rollout_count=240,
            critical_phase_queries=1200,
            chunk_size=1200,
        )
    dataset = config["dataset"]
    recovery_count = int(dataset["recovery_queries"])
    rollout_count = int(dataset["short_rollout_count"])
    rollout_steps = int(dataset["short_rollout_steps"])
    critical_count = int(dataset["critical_phase_queries"])
    if min(recovery_count, rollout_count, rollout_steps, critical_count) < 1:
        raise ValueError("all extension section sizes must be positive")
    total = recovery_count + rollout_count * rollout_steps + critical_count
    chunk_size = int(dataset["chunk_size"])
    if chunk_size < 1 or chunk_size % rollout_steps:
        raise ValueError("dataset.chunk_size must be a positive multiple of short_rollout_steps")

    output = _resolve(dataset["root"], base)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    signature = _hash(config)
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; use --resume")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != signature:
            raise ValueError("resume config differs from recorded hybrid extension config")
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "strategy": "hybrid_10ms_phase_consistent_acado_extensions_v1",
            "paper_core": False,
            "relationship": "augmentation for the immutable 40k AirSim/ACADO core",
            "input_contract": TRAJECTORY_INPUT_CONTRACT,
            "input_dimensions": 12,
            "input_features": list(TRAJECTORY_INPUT_FEATURES),
            "output_dimensions": 4,
            "output_features": list(OUTPUT_FEATURES),
            "config_sha256": signature,
            "source_dataset": str(_resolve(config["source"]["dataset"], base)),
            "completed_chunks": [],
            "section_counts": {
                "recovery_queries": recovery_count,
                "short_rollout_samples": rollout_count * rollout_steps,
                "critical_phase_queries": critical_count,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(manifest_path, manifest)

    source_path = _resolve(config["source"]["dataset"], base)
    environment_path = _resolve(config["source"]["environments"], base)
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"X", "U", "episode_id", "initial_state_id", "environment_id", "step_id", "target_gate_id", "position", "quaternion", "velocity", "acceleration", "reference_position", "reference_orientation"}
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"source core lacks arrays: {missing}")
    if source["X"].shape != (40000, 12) or source["U"].shape != (40000, 4):
        raise ValueError("hybrid extensions require a completed 40,000-row core")

    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    environments = _load_gates(environment_path)
    trajectories = _build_trajectories(
        source, environments, planner,
        start_velocity_mode=str(config["trajectory"]["start_velocity_mode"]).lower(),
        nominal_speed=float(config["planner"]["nominal_speed"]),
        final_exit_distance=float(config["trajectory"]["final_exit_distance"]),
    )
    reconstruction_error = _validate_reconstruction(
        source, trajectories, control_dt=float(config["timing"]["dt"]),
        tolerance=float(config["source"]["reconstruction_tolerance"]),
    )
    all_pool, low_pool, critical_pool = _build_pools(source, environments)
    controller = _controller(config, config_file)
    dt = float(config["timing"]["dt"])
    prediction_dt = float(config["mpc"]["prediction_dt"])
    horizon_nodes = int(config["mpc"]["horizon"]) + 1
    gravity = float(config["mpc"]["gravity"])
    max_speed = float(config["sampling"]["maximum_total_speed"])
    max_acceleration = float(config["sampling"]["maximum_total_acceleration"])
    families = list(config["sampling"]["families"])
    if not families:
        raise ValueError("sampling.families must not be empty")
    lower = np.array([float(config["mpc"]["min_thrust"]), -float(config["mpc"]["max_body_rate_xy"]), -float(config["mpc"]["max_body_rate_xy"]), -float(config["mpc"]["max_body_rate_z"])])
    upper = np.array([float(config["mpc"]["max_thrust"]), float(config["mpc"]["max_body_rate_xy"]), float(config["mpc"]["max_body_rate_xy"]), float(config["mpc"]["max_body_rate_z"])])
    max_attempts = int(config["sampling"]["maximum_attempts_per_candidate"])
    total_chunks = (total + chunk_size - 1) // chunk_size
    completed = sorted(int(value) for value in manifest.get("completed_chunks", []))
    sampling = config["sampling"]
    environment_balanced_buckets = bool(sampling.get("environment_balanced_buckets", False))
    normal_pattern = _bucket_pattern(
        sampling["control_bucket_slots"], allowed=(0, 1, 2),
    )
    rollout_pattern = _bucket_pattern(
        sampling["short_rollout_control_bucket_slots"], allowed=(0, 1),
    )
    family_index_by_name = {str(family["name"]): index for index, family in enumerate(families)}
    if len(family_index_by_name) != len(families):
        raise ValueError("sampling.families contains duplicate names")

    def indexes_for_names(names: Any) -> tuple[int, ...]:
        indexes = tuple(family_index_by_name[str(name)] for name in names)
        if not indexes:
            raise ValueError("a control-bucket family list must not be empty")
        return indexes

    eligible_by_bucket = {
        bucket: indexes_for_names(sampling["control_bucket_eligible_families"].get(name, ()))
        for bucket, name in enumerate(CONTROL_BUCKET_NAMES)
    }
    rollout_eligible_by_bucket = {
        bucket: indexes_for_names(sampling["short_rollout_eligible_families"].get(name, ()))
        for bucket, name in enumerate(CONTROL_BUCKET_NAMES[:2])
    }
    saturation_tolerance = float(sampling.get("saturation_tolerance", 1e-3))
    near_limit_fraction = float(sampling.get("near_limit_fraction", 0.15))

    def choose_base(rng: np.random.Generator, *, environment_id: int, pool: Mapping[int, np.ndarray], max_step: int | None = None) -> int:
        candidates = pool[environment_id]
        if max_step is not None:
            candidates = candidates[source["step_id"][candidates] <= max_step]
        if candidates.size == 0:
            candidates = all_pool[environment_id]
        return int(rng.choice(candidates))

    def make_state(rng: np.random.Generator, base_index: int, family_index: int) -> RawDroneState:
        env = int(source["environment_id"][base_index])
        gate_id = int(source["target_gate_id"][base_index])
        gate = environments[env][gate_id - 1]
        state, *_ = _perturb_state(
            source, base_index, gate, families[family_index], rng,
            maximum_speed=max_speed, maximum_acceleration=max_acceleration,
        )
        if str(families[family_index]["name"]) == "launch_recovery":
            state = _state_with_launch_speed(state, rng)
        return state

    def solve_one(state: RawDroneState, base_index: int, phase_step: int) -> tuple[np.ndarray, np.ndarray, Any, float, int]:
        trajectory = trajectories[int(source["episode_id"][base_index])]
        elapsed = (int(source["step_id"][base_index]) + phase_step) * dt
        references = planner.reference_window_for_trajectory(
            trajectory, elapsed=elapsed, dt=prediction_dt, horizon=horizon_nodes,
        )
        x = build_trajectory_input(state, references[0])
        started = time.perf_counter()
        u = np.asarray(controller.solve(state, references), dtype=np.float64)
        elapsed_solve = time.perf_counter() - started
        if u.shape != (4,) or not np.isfinite(u).all():
            raise RuntimeError("ACADO returned invalid control")
        bucket = classify_control_bucket(
            u, lower, upper, saturation_tolerance=saturation_tolerance,
            near_limit_fraction=near_limit_fraction,
        )
        return x, u, references[0], elapsed_solve, bucket

    for chunk_id in range(total_chunks):
        start = chunk_id * chunk_size
        stop = min(total, start + chunk_size)
        expected = stop - start
        chunk_path = output / "chunks" / f"chunk_{chunk_id:05d}.npz"
        if chunk_id in completed and chunk_path.exists():
            with np.load(chunk_path, allow_pickle=False) as archive:
                if archive["X"].shape == (expected, 12) and archive["U"].shape == (expected, 4):
                    continue
        rng = np.random.default_rng(int(config["random_seed"]) + 104729 * chunk_id)
        rows: dict[str, list[Any]] = {key: [] for key in (
            "X", "U", "sample_id", "query_group_id", "rollout_step", "source_kind", "sample_family", "control_bucket",
            "episode_id", "initial_state_id", "environment_id", "step_id", "target_gate_id", "position", "quaternion",
            "velocity", "acceleration", "reference_position", "reference_orientation", "reference_speed", "mpc_solve_time", "data_source",
        )}
        global_index = start
        while global_index < stop:
            if global_index < recovery_count:
                kind, pool, length = SOURCE_RECOVERY, None, 1
                relative = global_index
            elif global_index < recovery_count + rollout_count * rollout_steps:
                kind, pool, length = SOURCE_SHORT_ROLLOUT, None, rollout_steps
                relative = global_index - recovery_count
                if relative % rollout_steps:
                    raise RuntimeError("short-rollout chunk alignment invariant failed")
            else:
                kind, pool, length = SOURCE_CRITICAL_PHASE, critical_pool, 1
                relative = global_index - recovery_count - rollout_count * rollout_steps
            if global_index + length > stop:
                raise RuntimeError("chunk boundary must not split a short rollout")
            group_number = relative // max(1, length)
            # V4 uses a nested allocation: every environment receives the
            # same 1000 recovery/critical queries and 200 rollout groups,
            # with the same bucket cycle inside each environment.  The older
            # global cycle paired particular environment IDs with particular
            # buckets, which made validation/test distributions inequivalent.
            environment_id = group_number % len(environments)
            schedule_index = (
                group_number // len(environments)
                if environment_balanced_buckets else group_number
            )
            if kind == SOURCE_SHORT_ROLLOUT:
                target_bucket = rollout_pattern[schedule_index % len(rollout_pattern)]
                family_candidates = rollout_eligible_by_bucket[target_bucket]
            else:
                target_bucket = normal_pattern[schedule_index % len(normal_pattern)]
                family_candidates = eligible_by_bucket[target_bucket]
                # Explicit low-speed launch coverage is useful, but such states
                # inherently call for a saturated recovery command at 10 m/s.
                # Keep them only in their assigned, small saturation bucket.
                if kind == SOURCE_RECOVERY and target_bucket == 2:
                    family_candidates = (family_index_by_name["launch_recovery"],)
                elif kind == SOURCE_CRITICAL_PHASE and target_bucket == 2:
                    family_candidates = tuple(
                        index for index in family_candidates
                        if str(families[index]["name"]) != "launch_recovery"
                    ) or family_candidates
            group_id = (kind * 10_000_000) + (relative // max(1, length))
            accepted = False
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    family_index = family_candidates[(schedule_index + attempt) % len(family_candidates)]
                    if kind == SOURCE_RECOVERY and str(families[family_index]["name"]) == "launch_recovery":
                        base_index = choose_base(rng, environment_id=environment_id, pool=low_pool)
                    elif kind == SOURCE_CRITICAL_PHASE:
                        base_index = choose_base(rng, environment_id=environment_id, pool=critical_pool)
                    else:
                        base_index = choose_base(rng, environment_id=environment_id, pool=all_pool, max_step=99 - length)
                    state = make_state(rng, base_index, family_index)
                    if not _finite_state(state, max_speed=max_speed, max_acceleration=max_acceleration):
                        raise RuntimeError("sampled state exceeds configured physical limits")
                    controller.reset()
                    candidate: list[tuple[np.ndarray, np.ndarray, RawDroneState, Any, float, int]] = []
                    current = state
                    for phase_step in range(length):
                        x, u, reference, solve_seconds, bucket = solve_one(current, base_index, phase_step)
                        candidate.append((x, u, current, reference, solve_seconds, bucket))
                        if phase_step + 1 < length:
                            current = propagate_state(current, u, dt=dt, gravity=gravity)
                            if not _finite_state(current, max_speed=max_speed, max_acceleration=max_acceleration):
                                raise RuntimeError("short rollout left configured state limits")
                    # Singles must exactly match their requested difficulty.
                    # A five-step group is selected by its initial state; every
                    # later action is retained as a true local rollout, but no
                    # saturated action is allowed in this non-saturated section.
                    if candidate[0][5] != target_bucket:
                        raise RuntimeError(
                            f"requested control bucket {CONTROL_BUCKET_NAMES[target_bucket]!r}, "
                            f"received {CONTROL_BUCKET_NAMES[candidate[0][5]]!r}"
                        )
                    if kind == SOURCE_SHORT_ROLLOUT and any(item[5] == 2 for item in candidate):
                        raise RuntimeError("short rollout contains a saturated control")
                    saturated_here = sum(item[5] == 2 for item in candidate)
                except Exception as exc:  # ACADO errors and infeasible sampled states are retried locally.
                    last_error = exc
                    continue
                for phase_step, (x, u, current, reference, solve_seconds, bucket) in enumerate(candidate):
                    _append(rows, x=x, u=u, state=current, source=source, base=base_index, group_id=group_id,
                            rollout_step=phase_step, source_kind=kind, family=family_index, bucket=bucket,
                            solve_seconds=solve_seconds, reference=reference)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f"could not create source={kind} sample/group={group_id} after {max_attempts} attempts: {last_error}")
            global_index += length
        arrays = {
            key: np.asarray(value, dtype=(np.float32 if key in {"X", "U", "position", "quaternion", "velocity", "acceleration", "reference_position", "reference_orientation", "reference_speed"} else np.float64 if key == "mpc_solve_time" else np.int64))
            for key, value in rows.items()
        }
        arrays["sample_id"] = np.arange(start, stop, dtype=np.int64)
        arrays["rollout_step"] = arrays["rollout_step"].astype(np.int8)
        arrays["source_kind"] = arrays["source_kind"].astype(np.int8)
        arrays["sample_family"] = arrays["sample_family"].astype(np.int8)
        arrays["control_bucket"] = arrays["control_bucket"].astype(np.int8)
        arrays["data_source"] = arrays["data_source"].astype(np.int8)
        _atomic_npz(chunk_path, arrays)
        completed = sorted(set(completed) | {chunk_id})
        manifest.update(completed_chunks=completed, completed_samples=stop, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_write_json(manifest_path, manifest)
        written_saturated = sum(
            int(np.count_nonzero(np.asarray(part) == 2))
            for part in [rows["control_bucket"]]
        )
        print(
            f"Hybrid extensions: {len(completed)}/{total_chunks} chunks | samples={stop}/{total} "
            f"| chunk_saturated={written_saturated}",
            flush=True,
        )

    merged: dict[str, list[np.ndarray]] = {}
    for chunk_id in range(total_chunks):
        with np.load(output / "chunks" / f"chunk_{chunk_id:05d}.npz", allow_pickle=False) as archive:
            for key in archive.files:
                merged.setdefault(key, []).append(np.asarray(archive[key]))
    final_arrays = {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}
    saturated_fraction = float(np.mean(final_arrays["control_bucket"] == 2))
    maximum_saturated_fraction = float(sampling["maximum_saturated_fraction"])
    if saturated_fraction > maximum_saturated_fraction + 1e-12:
        raise RuntimeError(
            f"extension saturation fraction {saturated_fraction:.6f} exceeds configured maximum "
            f"{maximum_saturated_fraction:.6f}"
        )
    final_path = output / "final" / str(dataset["output_name"])
    _atomic_npz(final_path, final_arrays)
    source_counts = {str(kind): int(np.count_nonzero(final_arrays["source_kind"] == kind)) for kind in (SOURCE_RECOVERY, SOURCE_SHORT_ROLLOUT, SOURCE_CRITICAL_PHASE)}
    manifest.update(
        status="completed", final_dataset=str(Path("final") / final_path.name), X_shape=list(final_arrays["X"].shape),
        U_shape=list(final_arrays["U"].shape), trajectory_reconstruction_max_abs=reconstruction_error,
        source_kind_counts=source_counts, saturated_fraction=saturated_fraction,
        environment_balanced_buckets=environment_balanced_buckets,
        family_counts={str(family["name"]): int(np.count_nonzero(final_arrays["sample_family"] == index)) for index, family in enumerate(families)},
        mean_mpc_solve_time_seconds=float(np.mean(final_arrays["mpc_solve_time"])), updated_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(manifest_path, manifest)
    print(f"Hybrid extension complete: X={final_arrays['X'].shape} U={final_arrays['U'].shape}", flush=True)
    return output


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "hybrid_extension_config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    generate(args.config, resume=args.resume, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
