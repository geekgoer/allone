"""Resumable, environment-balanced collection for MPC-complete 300-D labels.

The collector never launches AirSim.  It reconstructs the immutable 40k
continuous trajectories and recomputes every label with a reset, converged
ACADO teacher.  Consequently feature ``X`` contains the whole online MPC
problem, while ``U`` has no dependency on the previous solver call.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from falcon_paper_fidelity.acado_mpc import AcadoMPCConfig
from falcon_paper_fidelity.mpc_complete_300d import (
    INPUT_CONTRACT,
    INPUT_DIMENSIONS,
    OUTPUT_CONTRACT,
    DeterministicAcadoTeacher,
    DeterministicTeacherConfig,
)
from falcon_paper_fidelity.motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner
from falcon_paper_fidelity.phase_recovery_dataset import _build_trajectories, _load_gates, _perturb_state, _validate_reconstruction
from falcon_paper_fidelity.short_rollout_dataset import propagate_state


CORE, RECOVERY, ROLLOUT = 0, 1, 2
BUCKET = {"interior": 0, "near_limit": 1, "saturated": 2}


def _resolve(value: str, base: Path) -> Path:
    return (base / value).resolve()


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _scaled_family(value: Mapping[str, Any], scale: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[key] = (np.asarray(item, dtype=np.float64) * scale).tolist() if key.endswith("_std") or key.endswith("_limit") else item
    return result


def _bucket(control: np.ndarray, lower: np.ndarray, upper: np.ndarray, tolerance: float, near_fraction: float) -> int:
    value = np.asarray(control, dtype=np.float64)
    if np.any(value <= lower + tolerance) or np.any(value >= upper - tolerance):
        return BUCKET["saturated"]
    if np.any(np.minimum(value - lower, upper - value) <= near_fraction * (upper - lower)):
        return BUCKET["near_limit"]
    return BUCKET["interior"]


def _valid_state(state: RawDroneState, *, max_speed: float, max_acceleration: float) -> bool:
    return (
        all(np.isfinite(value).all() for value in (state.position, state.quaternion, state.velocity, state.acceleration))
        and np.linalg.norm(state.velocity) <= max_speed + 1e-8
        and np.linalg.norm(state.acceleration) <= max_acceleration + 1e-8
        and -10.0 <= state.position[0] <= 70.0 and -25.0 <= state.position[1] <= 25.0 and -15.0 <= state.position[2] <= 1.0
    )


def _raw(source: Mapping[str, np.ndarray], index: int) -> RawDroneState:
    return RawDroneState(
        position=np.asarray(source["position"][index], dtype=np.float64).copy(),
        quaternion=np.asarray(source["quaternion"][index], dtype=np.float64).copy(),
        velocity=np.asarray(source["velocity"][index], dtype=np.float64).copy(),
        acceleration=np.asarray(source["acceleration"][index], dtype=np.float64).copy(),
        angular_velocity=np.zeros(3),
    )


def _append(rows: dict[str, list[Any]], *, sample_id: int, source_kind: int, family: int, group_id: int, rollout_step: int, saturation_allowed: bool, base_index: int, state: RawDroneState, reference: Any, result: Any, bucket: int) -> None:
    rows["X"].append(result.feature)
    rows["U"].append(result.control)
    rows["sample_id"].append(sample_id)
    rows["source_kind"].append(source_kind)
    rows["sample_family"].append(family)
    rows["query_group_id"].append(group_id)
    rows["rollout_step"].append(rollout_step)
    rows["saturation_allowed"].append(int(saturation_allowed))
    rows["control_bucket"].append(bucket)
    rows["base_row_id"].append(base_index)
    rows["episode_id"].append(-1)  # overwritten by the caller below
    rows["environment_id"].append(-1)
    rows["base_step_id"].append(-1)
    rows["target_gate_id"].append(-1)
    rows["position"].append(state.position)
    rows["quaternion"].append(state.quaternion)
    rows["velocity"].append(state.velocity)
    rows["acceleration"].append(state.acceleration)
    rows["reference_position"].append(reference.position)
    rows["reference_orientation"].append(reference.orientation)
    rows["reference_velocity"].append(reference.velocity)
    rows["reference_acceleration"].append(reference.acceleration)
    rows["teacher_iterations"].append(result.iterations)
    rows["teacher_final_delta"].append(result.final_control_delta)


def collect(config_path: Path, *, resume: bool) -> Path:
    config_file = config_path.resolve()
    base = config_file.parent
    config = json.loads(config_file.read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    planned = config["planned_dataset"]["samples_per_environment"]
    total = int(config["planned_dataset"]["total_samples"])
    if total != 40 * sum(int(value) for value in planned.values()):
        raise ValueError("planned_dataset.total_samples does not equal 40 environment quotas")
    if int(planned["short_rollout"]) % int(config["planned_dataset"]["rollout_steps"]) != 0:
        raise ValueError("short_rollout quota must divide exactly into rollout steps")
    chunk_size = int(dataset_cfg["chunk_size"])
    if chunk_size % int(config["planned_dataset"]["rollout_steps"]) != 0:
        raise ValueError("dataset.chunk_size must not split a short rollout")
    output = _resolve(dataset_cfg["root"], base)
    manifest_path = output / "manifest.json"
    signature = _hash(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != signature:
            raise RuntimeError("existing 300-D dataset has a different config; choose a new output root")
        if manifest.get("status") == "completed":
            print(f"Dataset already complete: {output}")
            return output / "final" / str(dataset_cfg["output_name"])
        if not resume:
            raise RuntimeError("unfinished collection exists; run with --resume")
    else:
        manifest = {
            "schema_version": 1, "status": "collecting", "created_at": datetime.now(timezone.utc).isoformat(),
            "config_file": str(config_file), "config_sha256": signature, "input_contract": INPUT_CONTRACT,
            "output_contract": OUTPUT_CONTRACT, "input_dimensions": INPUT_DIMENSIONS, "output_dimensions": 4,
            "total_samples": total, "completed_chunks": [], "completed_samples": 0,
            "sources": {"0": "recomputed_continuous_core", "1": "independent_recovery", "2": "short_closed_loop_rollout"},
            "environment_quota": {str(key): int(value) for key, value in planned.items()},
        }
        _atomic_json(manifest_path, manifest)

    source_path = _resolve(config["source"]["core_dataset"], base)
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    if source["X"].shape != (40000, 12):
        raise ValueError("collector expects the immutable 40,000-row core source")
    planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig(**config["planner"]))
    environments = _load_gates(_resolve(config["source"]["environments"], base))
    trajectories = _build_trajectories(source, environments, planner, start_velocity_mode=config["trajectory"]["start_velocity_mode"], nominal_speed=float(config["planner"]["nominal_speed"]), final_exit_distance=float(config["trajectory"]["final_exit_distance"]))
    reconstruction = _validate_reconstruction(source, trajectories, control_dt=float(config["timing"]["control_dt"]), tolerance=float(config["source"]["reconstruction_tolerance"]))
    mpc = dict(config["mpc"])
    mpc["library"] = str(_resolve(mpc["library"], base))
    teacher = DeterministicAcadoTeacher(AcadoMPCConfig(**mpc), DeterministicTeacherConfig(**{key: config["teacher"][key] for key in ("minimum_refinement_steps", "maximum_refinement_steps", "convergence_tolerance")}))
    lower = np.array([mpc["min_thrust"], -mpc["max_body_rate_xy"], -mpc["max_body_rate_xy"], -mpc["max_body_rate_z"]], dtype=np.float64)
    upper = np.array([mpc["max_thrust"], mpc["max_body_rate_xy"], mpc["max_body_rate_xy"], mpc["max_body_rate_z"]], dtype=np.float64)
    families = [_scaled_family(item, float(config["recovery_sampling"]["perturbation_scale"])) for item in config["families"]]
    dt, prediction_dt, horizon = float(config["timing"]["control_dt"]), float(config["timing"]["prediction_dt"]), int(config["timing"]["horizon"])
    max_speed, max_acceleration = float(config["planner"]["maximum_speed"]), float(config["planner"]["maximum_acceleration"])
    all_by_env = {env: np.flatnonzero(source["environment_id"] == env) for env in range(40)}
    rollout_by_env = {env: rows[source["step_id"][rows] <= 95] for env, rows in all_by_env.items()}
    core_per_env, recovery_per_env, rollout_per_env = (int(planned[name]) for name in ("core", "independent_recovery", "short_rollout"))
    rollout_steps = int(config["planned_dataset"]["rollout_steps"])
    recovery_begin = 40 * core_per_env
    rollout_begin = recovery_begin + 40 * recovery_per_env
    max_attempts = int(dataset_cfg["maximum_attempts_per_query"])
    period = int(dataset_cfg["extension_saturation_slot_period"])
    policy = config["saturation_policy"]
    completed = {int(value) for value in manifest.get("completed_chunks", [])}
    total_chunks = (total + chunk_size - 1) // chunk_size

    for chunk_id in range(total_chunks):
        start, stop = chunk_id * chunk_size, min(total, (chunk_id + 1) * chunk_size)
        path = output / "chunks" / f"chunk_{chunk_id:05d}.npz"
        if chunk_id in completed and path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if archive["X"].shape == (stop - start, INPUT_DIMENSIONS) and archive["U"].shape == (stop - start, 4):
                    continue
        rows: dict[str, list[Any]] = {name: [] for name in ("X", "U", "sample_id", "source_kind", "sample_family", "query_group_id", "rollout_step", "saturation_allowed", "control_bucket", "base_row_id", "episode_id", "environment_id", "base_step_id", "target_gate_id", "position", "quaternion", "velocity", "acceleration", "reference_position", "reference_orientation", "reference_velocity", "reference_acceleration", "teacher_iterations", "teacher_final_delta")}
        index = start
        while index < stop:
            if index < recovery_begin:
                kind, relative, length = CORE, index, 1
            elif index < rollout_begin:
                kind, relative, length = RECOVERY, index - recovery_begin, 1
            else:
                kind, relative, length = ROLLOUT, index - rollout_begin, rollout_steps
            if index + length > stop:
                raise RuntimeError("chunk boundary split a rollout")
            group = relative // length
            environment_id = group % 40
            slot = group // 40
            saturation_allowed = kind == CORE or (slot % period == period - 1)
            accepted = False
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                rng = np.random.default_rng(int(config["random_seed"]) + 1_000_003 * index + 65_537 * attempt)
                try:
                    candidates = all_by_env[environment_id] if kind != ROLLOUT else rollout_by_env[environment_id]
                    if kind == CORE:
                        base_index = int(candidates[(slot + attempt * 127) % candidates.size])
                        state = _raw(source, base_index)
                        family = -1
                    else:
                        base_index = int(candidates[(slot * 97 + attempt * 127) % candidates.size])
                        gate_id = int(source["target_gate_id"][base_index])
                        gate = next(item for item in environments[environment_id] if item.index == gate_id)
                        family = slot % len(families)
                        state, *_ = _perturb_state(source, base_index, gate, families[family], rng, maximum_speed=max_speed, maximum_acceleration=max_acceleration)
                    if not _valid_state(state, max_speed=max_speed, max_acceleration=max_acceleration):
                        raise RuntimeError("sampled state left configured physical bounds")
                    candidate: list[tuple[RawDroneState, Any, Any, int]] = []
                    current = state
                    for rollout_step in range(length):
                        elapsed = (int(source["step_id"][base_index]) + rollout_step) * dt
                        refs = planner.reference_window_for_trajectory(trajectories[int(source["episode_id"][base_index])], elapsed=elapsed, dt=prediction_dt, horizon=horizon + 1)
                        result = teacher.query(current, refs)
                        if not result.converged:
                            raise RuntimeError(f"teacher did not converge; delta={result.final_control_delta:.3e}")
                        bucket = _bucket(result.control, lower, upper, float(policy["absolute_tolerance"]), float(policy["near_limit_fraction"]))
                        candidate.append((current, refs[0], result, bucket))
                        if rollout_step + 1 < length:
                            current = propagate_state(current, result.control, dt=dt, gravity=float(config["planner"]["gravity"]))
                            if not _valid_state(current, max_speed=max_speed, max_acceleration=max_acceleration):
                                raise RuntimeError("short rollout left configured physical bounds")
                    if not saturation_allowed and any(item[3] == BUCKET["saturated"] for item in candidate):
                        raise RuntimeError("saturated extension outside its allowed slot")
                except Exception as error:
                    last_error = error
                    continue
                for rollout_step, (current, reference, result, bucket) in enumerate(candidate):
                    _append(rows, sample_id=index + rollout_step, source_kind=kind, family=family, group_id=(kind * 10_000_000 + group), rollout_step=rollout_step, saturation_allowed=saturation_allowed, base_index=base_index, state=current, reference=reference, result=result, bucket=bucket)
                    for name in ("episode_id", "environment_id", "base_step_id", "target_gate_id"):
                        rows[name][-1] = int(source[{"episode_id": "episode_id", "environment_id": "environment_id", "base_step_id": "step_id", "target_gate_id": "target_gate_id"}[name]][base_index])
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f"could not form source={kind} group={group} after {max_attempts} attempts: {last_error}")
            index += length
        arrays = {name: np.asarray(value, dtype=np.float32 if name in {"X", "U", "position", "quaternion", "velocity", "acceleration", "reference_position", "reference_orientation", "reference_velocity", "reference_acceleration", "teacher_final_delta"} else np.int8 if name in {"source_kind", "sample_family", "rollout_step", "saturation_allowed", "control_bucket", "teacher_iterations"} else np.int64) for name, value in rows.items()}
        _atomic_npz(path, arrays)
        completed.add(chunk_id)
        manifest.update(completed_chunks=sorted(completed), completed_samples=stop, trajectory_reconstruction_max_abs=reconstruction, updated_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, manifest)
        saturated = int(np.count_nonzero(arrays["control_bucket"] == BUCKET["saturated"]))
        print(f"300-D collection: {len(completed)}/{total_chunks} chunks | samples={stop}/{total} | chunk_saturated={saturated}", flush=True)

    merged: dict[str, list[np.ndarray]] = {}
    for chunk_id in range(total_chunks):
        with np.load(output / "chunks" / f"chunk_{chunk_id:05d}.npz", allow_pickle=False) as archive:
            for name in archive.files:
                merged.setdefault(name, []).append(np.asarray(archive[name]))
    final = {name: np.concatenate(parts, axis=0) for name, parts in merged.items()}
    if final["X"].shape != (total, INPUT_DIMENSIONS) or final["U"].shape != (total, 4):
        raise RuntimeError("final merged arrays have the wrong shape")
    final_path = output / "final" / str(dataset_cfg["output_name"])
    _atomic_npz(final_path, final)
    manifest.update(status="completed", completed_samples=total, completed_at=datetime.now(timezone.utc).isoformat(), final_dataset=str(final_path), source_kind_counts={str(kind): int(np.count_nonzero(final["source_kind"] == kind)) for kind in (CORE, RECOVERY, ROLLOUT)}, saturated_fraction=float(np.mean(final["control_bucket"] == BUCKET["saturated"])))
    _atomic_json(manifest_path, manifest)
    print(f"300-D dataset complete: X={final['X'].shape} U={final['U'].shape} saturated={manifest['saturated_fraction']:.3%}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("dataset_config.json")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="collect a 360-row disposable dataset across all 40 environments")
    args = parser.parse_args()
    if not args.smoke_test:
        collect(Path(args.config), resume=args.resume)
        return
    # Keep the smoke path explicit and isolated.  It exercises chunk resume,
    # all three source kinds, 300-D serialization, and five-step rollouts.
    original = Path(args.config).resolve()
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["planned_dataset"]["samples_per_environment"] = {"core": 2, "independent_recovery": 2, "short_rollout": 5}
    payload["planned_dataset"]["total_samples"] = 360
    payload["dataset"]["root"] = str(payload["dataset"]["root"]) + "_smoke"
    payload["dataset"]["output_name"] = "falcon_mpc_complete_300d_smoke.npz"
    payload["dataset"]["chunk_size"] = 200
    smoke = original.with_name(".dataset_config_smoke_300d.json")
    try:
        smoke.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        collect(smoke, resume=args.resume)
    finally:
        if smoke.exists():
            smoke.unlink()


if __name__ == "__main__":
    main()
