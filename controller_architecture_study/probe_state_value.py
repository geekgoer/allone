"""Paired, per-state analysis of completed 237D controllers; Validation only."""

import argparse
import csv
import itertools
import json

import numpy as np

from .decision_dataset import FAMILIES, atomic_json, atomic_npz
from .dynamic_probe_common import DATA, RUNS, ModelPool, new_output, provenance
from .eval_offline import error_metrics


def cluster_interval(delta, environments, repeats=2000, seed=20261004):
    """Paired environment bootstrap, not an IID bootstrap over correlated rows."""
    ids = np.unique(environments)
    if len(ids) < 2:
        return None
    sums = np.array([delta[environments == i].sum() for i in ids])
    counts = np.array([np.sum(environments == i) for i in ids])
    picks = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def paired_summary(errors, names, metadata):
    result = []
    families, environments = metadata["family_id"], metadata["environment_id"]
    for i, j in itertools.combinations(range(len(names)), 2):
        delta = errors[:, i] - errors[:, j]  # Positive means the second model helps.
        tail = errors[:, i] >= np.percentile(errors[:, i], 95)
        result.append({"first": names[i], "second": names[j], "samples": len(delta),
            "mean_error_reduction": float(delta.mean()),
            "second_wins_fraction": float(np.mean(delta > 0)),
            "second_worsens_fraction": float(np.mean(delta < 0)),
            "reduction_p05_p50_p95": np.percentile(delta, [5, 50, 95]).tolist(),
            "first_model_worst_5pct_mean_reduction": float(delta[tail].mean()),
            "environment_bootstrap_95pct_interval": cluster_interval(delta, environments),
            "environment_clusters": int(len(np.unique(environments))),
            "by_family": {FAMILIES[int(f)]: {"samples": int(np.sum(families == f)),
                "mean_error_reduction": float(delta[families == f].mean()),
                "second_wins_fraction": float(np.mean(delta[families == f] > 0))}
                for f in np.unique(families)}})
    return result


def exploratory_frontier(errors, names, costs):
    """Label-aware oracle is an upper bound, never a deployable selector."""
    budgets = np.linspace(min(costs), max(costs), 9)
    # Units: normalized squared error plus lambda * measured service milliseconds.
    lambdas = np.r_[0., np.logspace(-7, 0, 65)]
    candidates = [(float(c), float(errors[:, i].mean())) for i, c in enumerate(costs)]
    for penalty in lambdas:
        pick = np.argmin(errors + penalty*np.asarray(costs)[None, :], axis=1)
        candidates.append((float(np.mean(np.asarray(costs)[pick])),
                           float(np.mean(errors[np.arange(len(errors)), pick]))))
    return [{"average_service_budget_ms": float(b),
             "best_static_mse": float(min(errors[:, i].mean() for i, c in enumerate(costs) if c <= b+1e-12)),
             "label_aware_oracle_mse": min((e for c, e in candidates if c <= b+1e-12), default=None)}
            for b in budgets]


def run(args):
    import torch

    names = args.models
    evidence = provenance(names, validation=True)
    if args.preflight:
        print(json.dumps({"status": "PASS", "models": names, **evidence}, indent=2))
        return
    torch.set_num_threads(1)
    with np.load(DATA / "validation.npz", allow_pickle=False) as data:
        count = len(data["X"])
        ids = np.arange(count)
        if args.limit is not None:
            # Spread smoke rows through environments/families; do not take only env48 startup.
            ids = np.linspace(0, count-1, min(args.limit, count), dtype=int)
        x, target = data["X"][ids].astype(np.float32), data["U"][ids].astype(np.float32)
        metadata = {k: data[k][ids] for k in ("row_id", "environment_id", "route_id", "family_id",
                                             "actual_speed", "position_error", "geometry_family")}
    path = new_output("alignment")
    atomic_json(path / "provenance.json", evidence)
    normalized_predictions, models = [], []
    norm = None
    for name in names:
        print(f"Aligned inference: {name}, rows={len(x)}, device={args.device}", flush=True)
        # Load one model at a time for offline analysis; not a switching/memory benchmark.
        pool = ModelPool([name], args.device)
        norm = pool.norm
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(x), args.batch_size):
                batch = (x[start:start+args.batch_size] - norm["x_mean"]) / norm["x_std"]
                outputs.append(pool.models[name](torch.from_numpy(batch).to(args.device)).cpu().numpy())
        predictions = np.concatenate(outputs)
        normalized_predictions.append(predictions)
        models.append({"name": name, "parameter_count": pool.parameter_bytes // 4,
                       **error_metrics(predictions, target, norm),
                       "by_family": {FAMILIES[int(f)]: error_metrics(predictions[metadata["family_id"] == f],
                           target[metadata["family_id"] == f], norm) for f in np.unique(metadata["family_id"])}})
        del pool
    predictions = np.stack(normalized_predictions, axis=1)
    truth = (target - norm["u_mean"]) / norm["u_std"]
    errors = np.mean((predictions - truth[:, None, :])**2, axis=2)
    physical = predictions*norm["u_std"] + norm["u_mean"]
    # Raw, unclipped outputs match the training/evaluation metric; the online adapter clips separately.
    atomic_npz(path / "aligned_predictions.npz", {"model_names": np.asarray(names), "U_teacher": target,
        "U_predicted": physical, "normalized_error_per_state": errors, **metadata})
    with (path / "per_state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = list(metadata)
        writer.writerow(keys + [f"{name}_normalized_mse" for name in names])
        for k in range(len(x)):
            writer.writerow([metadata[key][k] for key in keys] + errors[k].tolist())
    bins = {}
    for field, boundaries in (("actual_speed", [0., 3., 5., 7., float("inf")]),
                              ("position_error", [0., .25, .5, 1., 2., float("inf")])):
        bins[field] = []
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            selected = (metadata[field] >= low) & (metadata[field] < high)
            bins[field].append({"range": f"[{low}, {high})", "samples": int(selected.sum()),
                               "mse": dict(zip(names, errors[selected].mean(axis=0).tolist())) if selected.any() else None})
    oracle_pick = errors.argmin(axis=1)
    best_static = int(errors.mean(axis=0).argmin())
    report = {"status": "COMPLETE", "exploratory_validation_only": True, "test_read": False,
        "samples": len(x), "smoke_limited": args.limit is not None, "models": models,
        "paired": paired_summary(errors, names, metadata), "state_bins": bins,
        "label_aware_oracle": {"not_deployable": True, "does_not_charge_compute_cost": True,
            "mse": float(errors.min(axis=1).mean()), "best_static_model": names[best_static],
            "best_static_mse": float(errors[:, best_static].mean()),
            "selection_counts": dict(zip(names, np.bincount(oracle_pick, minlength=len(names)).tolist()))},
        "limitations": ["Validation already selected checkpoints; exploratory, not an independent final test.",
                        "Only six held-out environments; cluster interval has limited precision.",
                        "Lower imitation MSE does not establish closed-loop benefit or safety."]}
    if args.timing_report:
        timing = json.loads(args.timing_report.read_text(encoding="utf-8"))
        for name in names:
            if timing["provenance"]["models"][name]["checkpoint_sha256"] != evidence["models"][name]["checkpoint_sha256"]:
                raise ValueError("timing and accuracy checkpoints differ")
            if not timing["per_model_service_ms"].get(name):
                raise ValueError("cost analysis needs a cycle replay containing all selected models")
        # One-stream replay isolates per-model service cost; total response also contains queueing.
        costs = [timing["per_model_service_ms"][n]["mean"] for n in names]
        report["exploratory_average_service_cost_frontier"] = exploratory_frontier(errors, names, costs)
    atomic_json(path / "report.json", report)
    print(json.dumps({"models": [{"name": m["name"], "normalized_mse": m["normalized_mse"]} for m in models],
                      "oracle": report["label_aware_oracle"]}, indent=2), flush=True)
    print(f"Report: {path / 'report.json'}", flush=True)


def main():
    from pathlib import Path
    from .dynamic_probe_common import DEFAULT_MODELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(RUNS), default=list(DEFAULT_MODELS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Smoke only; default evaluates all 9600 Validation rows")
    parser.add_argument("--timing-report", type=Path, help="Optional matching replay report for a label-aware cost upper bound")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size/limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
