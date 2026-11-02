"""Bounded CPU audit of frozen convex-teacher models; no training or Test access."""

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from airsim_interface import RawDroneState
from state_adapter import normalize_quaternion
from .convex_dataset import EXPERIMENT, SEED, sample_problem
from .convex_teacher import ConvexTeacher
from .decision_dataset import FAMILIES, atomic_json, build_features, sha256
from .eval_offline import load_checkpoint, load_normalization, OUTPUT_NAMES
from .fly_decision_dnn import StoredRoutePlanner


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def predict(torch, model, x, norm):
    batches = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            values = ((x[start:start+256]-norm["x_mean"])/norm["x_std"]).astype(np.float32)
            batches.append(model(torch.from_numpy(values)).numpy())
    return np.concatenate(batches)


def errors(predicted, data, norm):
    squared = (predicted-(data["U"]-norm["u_mean"])/norm["u_std"])**2
    mse = squared.mean(1)
    return {"samples": len(mse), "normalized_mse": float(mse.mean()),
        "physical_rmse_per_output": dict(zip(OUTPUT_NAMES, np.sqrt(squared.mean(0))*norm["u_std"])),
        "error_share_per_output": dict(zip(OUTPUT_NAMES, squared.sum(0)/squared.sum())),
        "top_1pct_share": float(np.sort(mse)[-max(1, len(mse)//100):].sum()/mse.sum()),
        "top_5pct_share": float(np.sort(mse)[-max(1, len(mse)//20):].sum()/mse.sum()),
        "families": {name: {"mse": float(mse[data["family_id"] == i].mean()),
            "share": float(mse[data["family_id"] == i].sum()/mse.sum())} for i, name in enumerate(FAMILIES)},
        "geometry_mse": {str(int(i)): float(mse[data["geometry_family"] == i].mean())
                          for i in np.unique(data["geometry_family"])},
        "route_mse_quantiles": np.quantile([mse[data["route_id"] == r].mean()
                                             for r in np.unique(data["route_id"])], [0, .5, .9, 1]).tolist()}


def numpy_json(value):
    if isinstance(value, dict):
        return {key: numpy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_json(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def stored_problem(data, i, planner):
    refs = planner.reference_window_for_trajectory(planner.trajectory,
        elapsed=float(data["phase_seconds"][i]), dt=.1, horizon=21)
    x = data["X"][i]
    state = RawDroneState(refs[0].position-x[10:13]*[1., -1., -1.],
        normalize_quaternion(x[3:7]*[1., 1., -1., -1.]),
        x[7:10]*[1., -1., -1.], np.zeros(3), np.zeros(3))
    return state, refs


def hard_label_probe():
    """Check 48 recovery labels without running model inference or changing data."""
    root = EXPERIMENT
    with np.load(root/"dnn_dataset/validation.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    manifest = read_json(root/"manifest.json")
    teacher = ConvexTeacher()
    if json.loads(json.dumps(teacher.signature())) != manifest["config"]["teacher"]:
        raise ValueError("teacher changed")
    rows, rng = [], np.random.default_rng(762)
    for family in (4, 5, 6, 7):
        ids = np.flatnonzero(data["family_id"] == family)
        for i in rng.choice(ids, 12, replace=False):
            name = f"env{data['environment_id'][i]:02d}_route{data['route_id'][i]%20:02d}"
            path = root/"routes"/f"{name}.json"
            if sha256(path) != manifest["chunks"][name]["route_sha256"]:
                raise ValueError("route changed")
            planner = StoredRoutePlanner(read_json(path))
            state, refs = stored_problem(data, i, planner)
            u, _ = teacher.query(state, refs)
            nearby, _ = teacher.query(replace(state, position=state.position+[.01, -.01, .01]), refs)
            repeat, _ = teacher.query(state, refs)
            unused, _ = teacher.query(state, [replace(ref, body_rates=ref.body_rates+[.1, -.1, .1]) for ref in refs])
            rows.append({"row_id": int(data["row_id"][i]), "family": FAMILIES[family],
                "label_max_delta": float(abs(u-data["U"][i]).max()),
                "repeat_max_delta": float(abs(u-repeat).max()),
                "unused_planner_rate_max_delta": float(abs(u-unused).max()),
                "neighbor_delta": abs(nearby-u).tolist()})
    result = {"test_read": False, "samples": len(rows), "rows": rows,
        "max_label_delta": max(row["label_max_delta"] for row in rows),
        "max_repeat_delta": max(row["repeat_max_delta"] for row in rows),
        "max_unused_rate_delta": max(row["unused_planner_rate_max_delta"] for row in rows),
        "max_neighbor_delta_per_output": np.max([row["neighbor_delta"] for row in rows], axis=0).tolist()}
    atomic_json(root/"diagnosis/convex_recovery_label_probes.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


def main():
    import torch
    torch.set_num_threads(2)
    root = EXPERIMENT
    report = {"test_read": False, "training_started": False, "models_or_datasets_modified": False,
              "models": [], "probe_note": "Fresh states are diagnostic only, never appended to any split."}
    destination = root/"diagnosis/convex_gap_audit.json"
    manifest = read_json(root/"manifest.json")
    split = read_json(root/"dnn_dataset/split_manifest.json")
    report["data_hashes_match"] = all(sha256(root/"dnn_dataset"/name) == split["artifact_sha256"][name]
                                     for name in ("train.npz", "validation.npz", "normalization.json"))
    if not report["data_hashes_match"]:
        raise ValueError("frozen data changed")
    norm = load_normalization(root/"dnn_dataset")
    data = {}
    for name in ("train", "validation"):
        with np.load(root/"dnn_dataset"/f"{name}.npz", allow_pickle=False) as archive:
            data[name] = {key: archive[key] for key in archive.files}
    train, val = data["train"], data["validation"]
    report["normalization_max_delta"] = {f"{prefix}_{stat}": float(np.max(abs(norm[f"{prefix}_{stat}"] -
        (train[key].astype(float).mean(0) if stat == "mean" else np.maximum(train[key].astype(float).std(0), 1e-6)))))
        for prefix, key in (("x", "X"), ("u", "U")) for stat in ("mean", "std")}
    report["split_overlap"] = {key: len(np.intersect1d(train[key], val[key]))
                              for key in ("row_id", "route_id", "environment_id")}
    normalized = {name: (d["X"]-norm["x_mean"])/norm["x_std"] for name, d in data.items()}
    unused = np.array([220+4*i+j for i in range(20) for j in (1, 2, 3)])
    report["input_distribution"] = {
        "constant_indices": np.flatnonzero(train["X"].std(0) < 1e-8).tolist(),
        "unused_planner_body_rate_features": unused.tolist(),
        "max_abs_mean_shift_std": float(abs(normalized["validation"].mean(0)).max()),
        "max_validation_abs_zscore": float(abs(normalized["validation"]).max()),
        "rows_outside_train_marginal_range": float(np.mean(np.any(
            (val["X"] < train["X"].min(0)-1e-5) | (val["X"] > train["X"].max(0)+1e-5), axis=1))),
        "label_std_ratio_val_train": (val["U"].std(0)/norm["u_std"]).tolist(),
        "train_routes": len(np.unique(train["route_id"])), "validation_routes": len(np.unique(val["route_id"])),
        "samples_per_route": 80, "samples_per_route_per_family": 10}
    selected = np.random.default_rng(471).choice(len(train["X"]), 4096, replace=False)
    eigenvalues = np.linalg.eigvalsh(np.cov(normalized["train"][selected].T))[::-1].clip(0)
    cumulative = np.cumsum(eigenvalues)/eigenvalues.sum()
    report["input_distribution"]["pca_components_95_99_999pct"] = [int(np.searchsorted(cumulative, p)+1)
                                                                                for p in (.95, .99, .999)]
    rows = read_json(root/"runs/comparison_scenic_falcon15.json")["models"]
    for row in rows:
        run = Path(row["run"])
        config = read_json(run/"config.json")
        expected_hashes = {name: split["artifact_sha256"][name]
                           for name in ("train.npz", "validation.npz", "normalization.json")}
        if config["dataset_file_sha256"] != expected_hashes:
            raise ValueError(f"model dataset provenance mismatch: {row['model']}")
        metrics = read_json(run/"offline_metrics.json")
        history = list(csv.DictReader((run/"train_log.csv").open(encoding="utf-8")))
        best = min(history, key=lambda item: float(item["validation_normalized_mse"]))
        before = min(float(item["validation_normalized_mse"]) for item in history[:300])
        record = {key: row[key] for key in ("model", "hidden_layers", "width", "parameter_count", "train_mse", "validation_mse", "best_epoch")}
        record.update(best_log_mse=float(best["validation_normalized_mse"]),
            cached_vs_best_log_delta=abs(metrics["validation"]["normalized_mse"]-float(best["validation_normalized_mse"])),
            last_100_validation_improvement_percent=100*(before-row["validation_mse"])/before,
            last_100_mean_grad_norm=float(np.mean([float(h["mean_grad_norm_before_clip"]) for h in history[-100:]])),
            last_100_clip_fraction=float(np.mean([float(h["clip_fraction"]) for h in history[-100:]])),
            per_family={name: {"train": metrics["train"]["by_family_id"][str(i)]["normalized_mse"],
                              "validation": metrics["validation"]["by_family_id"][str(i)]["normalized_mse"]}
                        for i, name in enumerate(FAMILIES)})
        report["models"].append(record)
    atomic_json(destination, numpy_json(report))
    print("Cached metrics, provenance and Train-only normalization audited.", flush=True)

    teacher = ConvexTeacher()
    if json.loads(json.dumps(teacher.signature())) != manifest["config"]["teacher"]:
        raise ValueError("teacher numeric version changed")
    fresh, route_selections, probes = {}, {}, []
    for name, envs in (("train", (0, 8, 16, 24, 32, 40)), ("validation", range(48, 54))):
        records, route_ids = [], []
        for env in envs:
            for route in range(4):
                route_name = f"env{env:02d}_route{route:02d}"
                path = root/"routes"/f"{route_name}.json"
                if sha256(path) != manifest["chunks"][route_name]["route_sha256"]:
                    raise ValueError("stored route changed")
                planner = StoredRoutePlanner(read_json(path))
                route_id = env*20+route
                route_ids.append(route_id)
                d = data[name]
                index = int(np.flatnonzero(d["route_id"] == route_id)[route % 8])
                state, refs = stored_problem(d, index, planner)
                control, _ = teacher.query(state, refs)
                neighbor, _ = teacher.query(replace(state, position=state.position+[.01, -.01, .01]), refs)
                probes.append({"split": name, "row_id": int(d["row_id"][index]),
                    "feature_max_delta": float(abs(build_features(state, refs)-d["X"][index]).max()),
                    "label_max_delta": float(abs(control-d["U"][index]).max()),
                    "neighbor_1cm_max_delta": float(abs(control-neighbor).max())})
                for family in range(8):
                    for repeat in range(2):
                        rng = np.random.default_rng([SEED, 9031, env, route, family, repeat])
                        state, refs, _ = sample_problem(planner.trajectory, planner, rng, family)
                        control, _ = teacher.query(state, refs)
                        records.append({"X": build_features(state, refs), "U": control, "family_id": family,
                                        "geometry_family": route % 4, "route_id": route_id})
        fresh[name] = {key: np.asarray([row[key] for row in records]) for key in records[0]}
        route_selections[name] = route_ids
        print(f"Fresh {name}-route diagnostic states: {len(records)}", flush=True)
    report["stored_label_probes"] = probes
    report["model_diagnostics"] = {}
    for model_name in ("plain_mlp_h4_w256_seed0", "plain_mlp_h6_w1024_seed0", "plain_mlp_h33_w512_seed0"):
        print(f"CPU inference audit: {model_name}", flush=True)
        model = load_checkpoint(root/"runs"/model_name/"best_model.pt", device="cpu")
        predictions = predict(torch, model, val["X"], norm)
        diagnostic = {"validation": errors(predictions, val, norm), "fresh_states": {}, "stored_matched_routes": {}}
        for name in ("train", "validation"):
            diagnostic["fresh_states"][name] = errors(predict(torch, model, fresh[name]["X"], norm), fresh[name], norm)
            mask = np.isin(data[name]["route_id"], route_selections[name])
            matched = {key: value[mask] for key, value in data[name].items()}
            diagnostic["stored_matched_routes"][name] = errors(predict(torch, model, matched["X"], norm), matched, norm)
        x = train["X"][selected[:256]].copy()
        base = predict(torch, model, x, norm)
        perturbed = x.copy()
        noise = np.random.default_rng(719).choice([-1., 1.], (len(x), len(unused)))
        perturbed[:, unused] += (.1*noise*norm["x_std"][unused]).astype(np.float32)
        changed = predict(torch, model, perturbed, norm)
        diagnostic["irrelevant_body_rate_perturbation"] = {
            "input_perturbation": "+/- 0.1 Train std on 60 planner rate coordinates; exact teacher is invariant",
            "prediction_normalized_mse_change": float(np.mean((changed-base)**2)),
            "prediction_physical_rmse_change": (np.sqrt(np.mean((changed-base)**2, 0))*norm["u_std"]).tolist()}
        # Inspect data gradients without optimizer steps or any weight mutation.
        model.zero_grad(set_to_none=True)
        batch = torch.from_numpy(((x-norm["x_mean"])/norm["x_std"]).astype(np.float32))
        target = torch.from_numpy(((train["U"][selected[:256]]-norm["u_mean"])/norm["u_std"]).astype(np.float32))
        loss = (model(batch)-target).square().mean()
        loss.backward()
        diagnostic["layer_gradients"] = [{"layer": name,
            "weight_norm": float(parameter.detach().norm()), "data_gradient_norm": float(parameter.grad.norm()),
            "weight_decay_gradient_norm": float(parameter.detach().norm())*1e-5}
            for name, parameter in model.named_parameters() if name.endswith("weight")]
        report["model_diagnostics"][model_name] = diagnostic
        atomic_json(destination, numpy_json(report))
        print(f"  stored Train routes={diagnostic['stored_matched_routes']['train']['normalized_mse']:.6g}; "
              f"fresh Train routes={diagnostic['fresh_states']['train']['normalized_mse']:.6g}; "
              f"Validation={diagnostic['validation']['normalized_mse']:.6g}", flush=True)
        del model
    print(f"Report: {destination}", flush=True)
    hard_label_probe()


if __name__ == "__main__":
    main()
