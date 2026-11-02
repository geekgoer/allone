"""Train an episode-level RBF SVR for the FALCON cost proxy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.svm import SVR


FEATURES = ["c_inverse_beta", "period_ms", "R_max_ms", "epsilon"]
TARGET = "J_proxy"


def _as_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(y_true), 1e-12)
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)
    return {
        "count": int(y_true.size),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0),
        "smape_percent": float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denominator) * 100.0),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fit_feature_maximum(x: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(x), axis=0)
    scale[scale <= 0.0] = 1.0
    return scale


def _fit_target_scale(y: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std <= 0.0:
        std = 1.0
    return mean, std


def _fit_svr(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    cv_splits: int,
    n_jobs: int,
) -> tuple[GridSearchCV, dict[str, Any]]:
    estimator = SVR(kernel="rbf")
    parameter_grid = {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "gamma": ["scale", 0.01, 0.1, 1.0, 10.0],
        "epsilon": [0.001, 0.01, 0.05, 0.1],
    }
    del seed
    del n_jobs
    splitter = GroupKFold(n_splits=cv_splits)
    search = GridSearchCV(
        estimator,
        parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x, y, groups=groups)
    return search, {
        "best_params": search.best_params_,
        "best_cv_rmse_scaled": float(-search.best_score_),
        "cv_splits": cv_splits,
        "grid_size": int(np.prod([len(values) for values in parameter_grid.values()])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    args = parser.parse_args()

    rows = _read_rows(args.dataset.resolve())
    rows = [row for row in rows if row.get("valid_for_svr", "").lower() == "true"]
    if not rows:
        raise ValueError("dataset contains no valid_for_svr=true rows")

    required = FEATURES + [TARGET, "source_directory", "record_id"]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")

    x = np.asarray([[ _as_float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([_as_float(row[TARGET]) for row in rows], dtype=np.float64)
    groups = np.asarray([row["source_directory"] for row in rows], dtype=object)
    group_names = sorted(set(groups.tolist()))
    if len(group_names) < 3:
        raise ValueError(f"need at least 3 source_directory groups, found {len(group_names)}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_index, test_index = next(splitter.split(x, y, groups=groups))
    train_groups = sorted(set(groups[train_index].tolist()))
    test_groups = sorted(set(groups[test_index].tolist()))
    if set(train_groups) & set(test_groups):
        raise AssertionError("group leakage detected between train and test")

    train_scale = _fit_feature_maximum(x[train_index])
    train_x = x[train_index] / train_scale
    test_x = x[test_index] / train_scale
    target_mean, target_std = _fit_target_scale(y[train_index])
    train_y = (y[train_index] - target_mean) / target_std

    cv_splits = min(args.cv_splits, len(train_groups))
    if cv_splits < 2:
        raise ValueError(f"need at least 2 training groups for cross-validation, found {len(train_groups)}")
    search, search_info = _fit_svr(
        train_x,
        train_y,
        groups[train_index],
        args.seed,
        cv_splits,
        -1,
    )
    holdout_model = search.best_estimator_
    test_pred = holdout_model.predict(test_x) * target_std + target_mean
    train_pred = holdout_model.predict(train_x) * target_std + target_mean

    final_scale = _fit_feature_maximum(x)
    final_x = x / final_scale
    final_mean, final_std = _fit_target_scale(y)
    final_y = (y - final_mean) / final_std
    final_model = SVR(kernel="rbf", **search.best_params_)
    final_model.fit(final_x, final_y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": holdout_model,
            "feature_names": FEATURES,
            "target_name": TARGET,
            "feature_maximum": train_scale.tolist(),
            "target_mean": target_mean,
            "target_std": target_std,
            "normalization": "x_j = x_j / max(abs(train_x_j)); y = (J_proxy-target_mean)/target_std",
            "best_params": search.best_params_,
        },
        args.output_dir / "svr_model_holdout.joblib",
    )
    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURES,
            "target_name": TARGET,
            "feature_maximum": final_scale.tolist(),
            "target_mean": final_mean,
            "target_std": final_std,
            "normalization": "x_j = x_j / max(abs(all_valid_x_j)); y = (J_proxy-target_mean)/target_std",
            "best_params": search.best_params_,
        },
        args.output_dir / "svr_model_all_data.joblib",
    )

    with (args.output_dir / "holdout_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "source_directory", "split", "J_proxy", "prediction", "absolute_error"],
        )
        writer.writeheader()
        for index, prediction in zip(train_index, train_pred):
            writer.writerow({
                "record_id": rows[index]["record_id"],
                "source_directory": rows[index]["source_directory"],
                "split": "train",
                "J_proxy": y[index],
                "prediction": prediction,
                "absolute_error": abs(y[index] - prediction),
            })
        for index, prediction in zip(test_index, test_pred):
            writer.writerow({
                "record_id": rows[index]["record_id"],
                "source_directory": rows[index]["source_directory"],
                "split": "test",
                "J_proxy": y[index],
                "prediction": prediction,
                "absolute_error": abs(y[index] - prediction),
            })

    _write_json(
        args.output_dir / "metrics.json",
        {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset.resolve()),
            "records_valid": len(rows),
            "features": FEATURES,
            "target": TARGET,
            "split_group": "source_directory",
            "seed": args.seed,
            "train_groups": train_groups,
            "test_groups": test_groups,
            "train_records": int(len(train_index)),
            "test_records": int(len(test_index)),
            "search": search_info,
            "train_metrics": _metrics(y[train_index], train_pred),
            "test_metrics": _metrics(y[test_index], test_pred),
        },
    )
    _write_json(
        args.output_dir / "train_manifest.json",
        {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset.resolve()),
            "records_valid": len(rows),
            "source_directories": group_names,
            "test_split_group_policy": "GroupShuffleSplit by source_directory; no source directory appears in both splits",
            "holdout_model": "svr_model_holdout.joblib",
            "deployment_model": "svr_model_all_data.joblib",
            "best_params": search.best_params_,
            "features": FEATURES,
            "target": TARGET,
            "target_definition": "J_proxy = 1 / tracking_cost; engineering proxy, not the undisclosed exact paper J",
            "R_definition": "R_max_ms measured per episode from state-to-command dispatch timestamps",
        },
    )
    print(json.dumps({
        "records_valid": len(rows),
        "train_records": len(train_index),
        "test_records": len(test_index),
        "train_groups": train_groups,
        "test_groups": test_groups,
        "best_params": search.best_params_,
        "test_metrics": _metrics(y[test_index], test_pred),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
