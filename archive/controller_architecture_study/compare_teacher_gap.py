"""Read-only 12D versus 300D gap audit; writes a report, never trains or flies.

The warm-start and changed-yaw-cost solves are diagnostic counterfactuals only.
Their outputs are NOT inserted into either teacher dataset.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCController
from state_adapter import normalize_quaternion
from .audit_capacity_20261002 import preclip_acceleration
from .decision_dataset import FAMILIES, ROOT, RefinedTeacher, atomic_json, build_features, local_problem, sha256
from .decision_dataset_96k import EXPERIMENT
from .eval_offline import load_checkpoint, load_normalization
from .fly_decision_dnn import StoredRoutePlanner


RECOVERY = EXPERIMENT.with_name("decision_recovery_v1")
OLD = ROOT / "experiments/paper_track_p_12d_5ms_v1"
DEFAULT_REPORT = RECOVERY / "diagnosis/linear_vs_nmpc_gap.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def old_formula(x, controller):
    """Exact algebraic reduction, not a regressor fitted to the labels."""
    intercept = preclip_acceleration(controller, np.zeros(12))
    coefficients = np.column_stack([preclip_acceleration(controller, unit) - intercept for unit in np.eye(12)])
    acceleration = np.clip(x @ coefficients.T + intercept, -controller.max_horizontal_accel,
                           controller.max_horizontal_accel)
    thrust_vector = acceleration.copy()
    thrust_vector[:, 2] -= controller.gravity
    thrust = np.clip(np.linalg.norm(thrust_vector, axis=1), 0, controller.max_thrust_accel)
    attitude = np.zeros((len(x), 3))
    attitude[:, 0] = acceleration[:, 1] / np.maximum(thrust, controller.gravity)
    attitude[:, 1] = -acceleration[:, 0] / np.maximum(thrust, controller.gravity)
    rates = np.clip(controller.attitude_gain * (np.clip(attitude, -.45, .45) + x[:, 3:6]),
                    -controller.max_body_rate, controller.max_body_rate)
    return np.column_stack((thrust, rates)), intercept, coefficients


def old_audit():
    controller = MPCController(dt=.05, horizon=20, max_horizontal_accel=6., max_thrust_accel=18., max_body_rate=3.)
    result = {"splits": {}, "historical_models": []}
    for split in ("train", "validation"):
        with np.load(OLD / "dnn_dataset" / f"{split}.npz", allow_pickle=False) as archive:
            x, u = archive["X"].astype(float), archive["U"].astype(float)
            if "gate_orientation" not in archive or not np.allclose(archive["gate_orientation"], [1., 0., 0., 0.]):
                raise ValueError("closed-form audit requires the original identity-orientation gates")
        reconstructed, intercept, coefficients = old_formula(x, controller)
        result["splits"][split] = {"samples": len(x), "formula_max_abs_error": float(abs(reconstructed-u).max()),
                                   "formula_physical_mse": float(np.mean((reconstructed-u)**2))}
    result.update(preclip_intercept=intercept.tolist(), preclip_coefficients=coefficients.tolist(),
                  inactive_input_indices=[0, 9, 10, 11],
                  note="Environment translation cancels; eight active coordinates, not eight equally complex dependencies.")
    previous = read_json(ROOT / "controller_architecture_study/results/capacity_finetune_v1/comparison.json")
    for row in previous["models"]:
        result["historical_models"].append({key: row[key] for key in ("model", "train_mse", "validation_mse")})
        result["historical_models"][-1]["validation_train_ratio"] = row["validation_mse"] / row["train_mse"]
    return result


def cached_gap():
    result = []
    for arm in ("old_control", "added_only", "added_regularized"):
        for width in (64, 128, 256, 1024):
            run = RECOVERY / "runs" / arm / f"plain_mlp_h4_w{width}_seed0"
            metrics = read_json(run / "offline_metrics.json")
            families = []
            for k, family in enumerate(FAMILIES):
                t = metrics["train"]["by_family_id"][str(k)]["normalized_mse"]
                v = metrics["validation"]["by_family_id"][str(k)]["normalized_mse"]
                families.append({"family": family, "train_mse": t, "validation_mse": v, "ratio": v/t})
            result.append({"arm": arm, "width": width, "train_mse": metrics["train"]["normalized_mse"],
                           "validation_mse": metrics["validation"]["normalized_mse"], "families": families,
                           "equal_family_train_mse": float(np.mean([f["train_mse"] for f in families]))})
    return result


def diagnostic_solve(teacher, state, refs, *, yaw_cost=1., cold=True, max_iterations=128):
    """Same compiled dynamics; never treat an unconverged solve as a label."""
    state, refs = local_problem(state, refs)
    controller = teacher.controller
    if cold:
        controller._raise_status(controller._dll.falcon_acado_initialize(), "diagnostic initialize")
        # initialize resets weights, so any counterfactual configuration goes AFTER it.
        arrays = [np.asarray(a, dtype=np.float64) for a in
                  ([200., 500., 50., 10., 0.], [1., 1., yaw_cost], [2., 20., 3., 2.])]
        pointer = ctypes.POINTER(ctypes.c_double)
        status = controller._dll.falcon_acado_configure(*[a.ctypes.data_as(pointer) for a in arrays])
        controller._raise_status(status, "diagnostic configure")
        controller.reset()
    previous, delta = None, float("inf")
    for iteration in range(1, max_iterations + 1):
        output = controller.solve(state, refs)
        if previous is not None:
            delta = float(np.max(abs(output-previous)))
        previous = output.copy()
        diagnostic = controller.last_diagnostics
        finite = np.isfinite(output).all() and np.isfinite(list(diagnostic.values())).all()
        bounded = np.all(output >= np.array([2., -3., -3., -2.])-1e-4) and np.all(output <= np.array([20., 3., 3., 2.])+1e-4)
        converged = bool(finite and bounded and iteration >= 4 and delta <= 1e-4 and abs(diagnostic["kkt"]) <= 1e-3)
        if converged:
            break
    return {"converged": converged, "output": output.tolist(), "iterations": iteration,
            "action_delta": delta, **diagnostic}


def reconstruct(data, index, source):
    name = f"env{data['environment_id'][index]:02d}_route{data['route_id'][index]%20:02d}"
    path = EXPERIMENT / "routes" / f"{name}.json"
    if sha256(path) != source["chunks"][name]["route_sha256"]:
        raise ValueError("route differs from recorded teacher source")
    planner = StoredRoutePlanner(read_json(path))
    refs = planner.reference_window_for_trajectory(planner.trajectory,
             elapsed=float(data["phase_seconds"][index]), dt=.1, horizon=21)
    x = data["X"][index]
    state = RawDroneState(refs[0].position - x[10:13]*[1., -1., -1.],
                         normalize_quaternion(x[3:7]*[1., 1., -1., -1.]),
                         x[7:10]*[1., -1., -1.], np.zeros(3), np.zeros(3))
    if np.max(abs(build_features(state, refs)-x)) > 1e-5:
        raise ValueError("feature reconstruction failed")
    return state, refs


def nmpc_probe():
    import torch

    torch.set_num_threads(2)
    source = read_json(EXPERIMENT / "manifest.json")
    teacher = RefinedTeacher()
    if sha256(teacher.controller.library_path) != source["config"]["dll_sha256"]:
        raise ValueError("teacher DLL differs from recorded source")
    checkpoint = RECOVERY / "runs/added_regularized/plain_mlp_h4_w1024_seed0/best_model.pt"
    model = load_checkpoint(checkpoint, device="cpu")
    norm = load_normalization(EXPERIMENT / "dnn_dataset")
    with np.load(EXPERIMENT / "dnn_dataset/validation.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data["X"]), 512):
            batch = (data["X"][start:start+512]-norm["x_mean"])/norm["x_std"]
            predictions.append(model(torch.from_numpy(batch)).numpy())
    predicted = np.concatenate(predictions)
    squared = (predicted-(data["U"]-norm["u_mean"])/norm["u_std"])**2
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "normalized_error_share_per_output": (squared.sum(0)/squared.sum()).tolist(),
              "u_std": norm["u_std"].tolist(), "rows": [],
              "note": "Selected probes establish counterexamples, not population failure rates or a new accepted teacher."}
    result["yaw_label_strata"] = {}
    for name, selected in (("abs_yaw_le_0_3", abs(data["U"][:, 3]) <= .3),
                           ("abs_yaw_gt_0_3", abs(data["U"][:, 3]) > .3),
                           ("yaw_at_bound", abs(data["U"][:, 3]) >= 1.99)):
        result["yaw_label_strata"][name] = {"samples": int(selected.sum()),
            "normalized_error_share": float(squared[selected].sum()/squared.sum()),
            "yaw_rmse_rad_s": float(np.sqrt(squared[selected, 3].mean())*norm["u_std"][3])}
    result["yaw_label_strata"] = {}
    for name, selected in (("abs_yaw_le_0_3", abs(data["U"][:, 3]) <= .3),
                           ("abs_yaw_gt_0_3", abs(data["U"][:, 3]) > .3),
                           ("yaw_at_bound", abs(data["U"][:, 3]) >= 1.99)):
        result["yaw_label_strata"][name] = {"samples": int(selected.sum()),
            "normalized_error_share": float(squared[selected].sum()/squared.sum()),
            "yaw_rmse_rad_s": float(np.sqrt(squared[selected, 3].mean())*norm["u_std"][3])}
    rng = np.random.default_rng(719)
    for family in (0, 7):
        ids = np.flatnonzero(data["family_id"] == family)
        for selection, indices in (("worst_four", ids[np.argsort(squared[ids].mean(1))[-4:]]),
                                   ("random_four", rng.choice(ids, 4, replace=False))):
            for index in indices:
                state, refs = reconstruct(data, index, source)
                perturbed = replace(state, position=state.position+np.array([.01, -.01, .01]))
                row = {"row_id": int(data["row_id"][index]), "family": FAMILIES[family],
                       "selection": selection, "stored_label": data["U"][index].tolist(),
                       "prediction": (predicted[index]*norm["u_std"]+norm["u_mean"]).tolist(), "cost_probes": []}
                for cost in (1., 5., 10.):
                    base = diagnostic_solve(teacher, state, refs, yaw_cost=cost)
                    nearby = diagnostic_solve(teacher, perturbed, refs, yaw_cost=cost)
                    # Same exact base problem, initialized from the neighboring solution.
                    warm = diagnostic_solve(teacher, state, refs, yaw_cost=cost, cold=False) if nearby["converged"] else None
                    row["cost_probes"].append({"yaw_cost": cost, "base_cold": base,
                                              "position_1cm_cold": nearby, "base_neighbor_warm": warm})
                result["rows"].append(row)
                print(f"audited {row['family']} row={row['row_id']} selection={selection}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = {"test_read": False, "training_started": False, "airsim_connected": False,
              "datasets_or_models_modified": False, "old_12d": old_audit(), "current_models": cached_gap(),
              "nmpc_probe": nmpc_probe()}
    atomic_json(args.report.resolve(), report)
    print(f"Report: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
