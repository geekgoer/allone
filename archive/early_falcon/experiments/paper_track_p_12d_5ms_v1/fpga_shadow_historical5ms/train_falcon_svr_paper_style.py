"""Train a resumable, grouped RBF-SVR for the paper-style FALCON metric."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.svm import SVR


TARGET = "J_paper_style"
FEATURE_NAMES = ["c", "period_ms", "R_max_ms", "epsilon"]
REQUIRED_COLUMNS = [
    "record_id",
    "source_directory",
    "beta_test_normalized_mse",
    "period_ms",
    "R_max_ms",
    "epsilon",
    TARGET,
    "valid_for_svr",
]
GRID = {
    "C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "gamma": ["scale", 0.01, 0.1, 1.0, 10.0],
    "epsilon": [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
}
CHECKPOINT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
    )


def _atomic_joblib_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        joblib.dump(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value: str, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {column} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {column} is not finite: {value!r}")
    return result


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_safe_episode(row: dict[str, str], columns: list[str]) -> bool:
    if not _is_true(row.get("valid_for_svr", "")):
        return False
    for name in ("completed", "pure_fpga_episode"):
        if name in columns and not _is_true(row.get(name, "")):
            return False
    for name in ("fallback_count", "deadline_miss_count", "timeout_count"):
        if name in columns and int(row.get(name, "0") or "0") != 0:
            return False
    for name in ("collision", "error"):
        if name in columns and (row.get(name) or "").strip():
            return False
    return True


def _read_dataset(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(
                "dataset is missing required columns: "
                + ", ".join(missing)
                + ". Existing J_proxy data cannot be renamed to J_paper_style."
            )
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not _is_safe_episode(row, columns):
                continue
            record_id = (row.get("record_id") or "").strip()
            source_directory = (row.get("source_directory") or "").strip()
            if not record_id or not source_directory:
                raise ValueError(f"row {row_number}: record_id and source_directory are required")
            beta = _float(row["beta_test_normalized_mse"], "beta_test_normalized_mse", row_number)
            if beta <= 0.0:
                raise ValueError(f"row {row_number}: beta_test_normalized_mse must be > 0")
            period_ms = _float(row["period_ms"], "period_ms", row_number)
            r_max_ms = _float(row["R_max_ms"], "R_max_ms", row_number)
            epsilon = _float(row["epsilon"], "epsilon", row_number)
            target = _float(row[TARGET], TARGET, row_number)
            if period_ms <= 0.0 or r_max_ms < 0.0:
                raise ValueError(f"row {row_number}: period_ms must be > 0 and R_max_ms >= 0")
            if target <= 0.0:
                raise ValueError(f"row {row_number}: {TARGET} must be > 0")
            rows.append(row)

    if not rows:
        raise ValueError("dataset contains no valid_for_svr=true rows")
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record_id must be unique among valid_for_svr=true rows")

    x = np.asarray(
        [
            [
                1.0 / _float(row["beta_test_normalized_mse"], "beta_test_normalized_mse", index),
                _float(row["period_ms"], "period_ms", index),
                _float(row["R_max_ms"], "R_max_ms", index),
                _float(row["epsilon"], "epsilon", index),
            ]
            for index, row in enumerate(rows, start=2)
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [_float(row[TARGET], TARGET, index) for index, row in enumerate(rows, start=2)],
        dtype=np.float64,
    )
    groups = np.asarray([row["source_directory"] for row in rows], dtype=object)
    return rows, x, y, groups


def _feature_scale(x: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(x[:, :3]), axis=0)
    scale[scale <= 0.0] = 1.0
    return scale


def _transform(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = np.asarray(x, dtype=np.float64).copy()
    transformed[:, :3] /= scale
    return transformed


def _params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _candidates() -> list[dict[str, Any]]:
    return [
        {"C": c, "gamma": gamma, "epsilon": epsilon}
        for c, gamma, epsilon in itertools.product(
            GRID["C"], GRID["gamma"], GRID["epsilon"]
        )
    ]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    denominator = np.maximum(np.abs(y_true), 1e-12)
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)
    return {
        "count": int(y_true.size),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0),
        "smape_percent": float(
            np.mean(2.0 * np.abs(y_true - y_pred) / smape_denominator) * 100.0
        ),
    }


def _fit_model(x: np.ndarray, y: np.ndarray, scale: np.ndarray, params: dict[str, Any]) -> SVR:
    model = SVR(kernel="rbf", **params)
    model.fit(_transform(x, scale), y)
    return model


def _evaluate_candidate(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict[str, Any],
    cv_splits: int,
) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=cv_splits)
    fold_rmses: list[float] = []
    for fold_train, fold_validation in splitter.split(x, y, groups=groups):
        scale = _feature_scale(x[fold_train])
        model = _fit_model(x[fold_train], y[fold_train], scale, params)
        prediction = model.predict(_transform(x[fold_validation], scale))
        residual = prediction - y[fold_validation]
        fold_rmses.append(float(math.sqrt(np.mean(residual * residual))))
    return {
        "params": params,
        "fold_rmse": fold_rmses,
        "mean_cv_rmse": float(np.mean(fold_rmses)),
        "std_cv_rmse": float(np.std(fold_rmses)),
        "completed_utc": _utc_now(),
    }


def _new_state(
    dataset: Path,
    dataset_hash: str,
    seed: int,
    test_size: float,
    cv_splits: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    train_groups = sorted(set(groups[train_index].tolist()))
    test_groups = sorted(set(groups[test_index].tolist()))
    return {
        "format_version": CHECKPOINT_VERSION,
        "status": "searching",
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "dataset": str(dataset),
        "dataset_sha256": dataset_hash,
        "seed": seed,
        "test_size": test_size,
        "cv_splits": cv_splits,
        "train_index": train_index.tolist(),
        "test_index": test_index.tolist(),
        "train_groups": train_groups,
        "test_groups": test_groups,
        "grid": GRID,
        "completed_candidates": {},
        "active_candidate_index": None,
        "best_candidate": None,
    }


def _validate_or_load_state(
    checkpoint: Path,
    dataset: Path,
    dataset_hash: str,
    seed: int,
    test_size: float,
    cv_splits: int,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        expected = {
            "format_version": CHECKPOINT_VERSION,
            "dataset_sha256": dataset_hash,
            "seed": seed,
            "test_size": test_size,
            "cv_splits": cv_splits,
            "grid": GRID,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"checkpoint mismatch for {key}: "
                    f"stored={state.get(key)!r}, requested={value!r}. "
                    "Use the original command or a new output directory."
                )
        train_index = np.asarray(state["train_index"], dtype=np.int64)
        test_index = np.asarray(state["test_index"], dtype=np.int64)
        if set(train_index.tolist()) & set(test_index.tolist()):
            raise ValueError("checkpoint contains train/test index overlap")
        if max(train_index.tolist() + test_index.tolist(), default=-1) >= len(x):
            raise ValueError("checkpoint indices exceed the current dataset")
        return state

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(x, y, groups=groups))
    train_group_set = set(groups[train_index].tolist())
    test_group_set = set(groups[test_index].tolist())
    if train_group_set & test_group_set:
        raise AssertionError("group leakage detected between train and test")
    if len(train_group_set) < cv_splits:
        raise ValueError(
            f"the grouped train split has {len(train_group_set)} groups, "
            f"but --cv-splits={cv_splits}; use a smaller CV split count or a different test size"
        )
    state = _new_state(
        dataset,
        dataset_hash,
        seed,
        test_size,
        cv_splits,
        train_index,
        test_index,
        groups,
    )
    _write_json(checkpoint, state)
    return state


def _write_cv_scores(path: Path, state: dict[str, Any]) -> None:
    fieldnames = [
        "candidate_index",
        "C",
        "gamma",
        "epsilon",
        "mean_cv_rmse",
        "std_cv_rmse",
        "fold_rmse",
        "completed_utc",
    ]
    rows: list[dict[str, Any]] = []
    completed = state.get("completed_candidates", {})
    for result in completed.values():
        params = result["params"]
        rows.append(
            {
                "candidate_index": result["candidate_index"],
                "C": params["C"],
                "gamma": params["gamma"],
                "epsilon": params["epsilon"],
                "mean_cv_rmse": result["mean_cv_rmse"],
                "std_cv_rmse": result["std_cv_rmse"],
                "fold_rmse": json.dumps(result["fold_rmse"]),
                "completed_utc": result["completed_utc"],
            }
        )
    rows.sort(key=lambda row: int(row["candidate_index"]))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _update_best(state: dict[str, Any]) -> None:
    completed = list(state.get("completed_candidates", {}).values())
    if completed:
        state["best_candidate"] = min(
            completed,
            key=lambda result: (result["mean_cv_rmse"], result["std_cv_rmse"]),
        )


def _save_state(checkpoint: Path, scores_path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc_now()
    _update_best(state)
    _write_json(checkpoint, state)
    _write_cv_scores(scores_path, state)


def _model_payload(
    model: SVR,
    scale: np.ndarray,
    params: dict[str, Any],
    dataset: Path,
    target_name: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "target_name": target_name,
        "feature_maximum_for_c_T_R": scale.tolist(),
        "epsilon_feature_is_unscaled": True,
        "target_is_unscaled": True,
        "normalization": "c'=c/c_max, T'=T/T_max, R'=R/R_max; epsilon unchanged",
        "c_definition": "c=1/beta_test_normalized_mse",
        "best_params": params,
        "dataset": str(dataset),
    }


def _write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    train_index: np.ndarray,
    test_index: np.ndarray,
    train_prediction: np.ndarray,
    test_prediction: np.ndarray,
) -> None:
    fieldnames = [
        "record_id",
        "source_directory",
        "split",
        TARGET,
        "prediction",
        "absolute_error",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, indices, predictions in (
            ("train", train_index, train_prediction),
            ("test", test_index, test_prediction),
        ):
            for index, prediction in zip(indices.tolist(), predictions.tolist()):
                actual = float(rows[index][TARGET])
                writer.writerow(
                    {
                        "record_id": rows[index]["record_id"],
                        "source_directory": rows[index]["source_directory"],
                        "split": split,
                        TARGET: actual,
                        "prediction": float(prediction),
                        "absolute_error": abs(actual - float(prediction)),
                    }
                )
    temporary.replace(path)


def _finalize(
    output_dir: Path,
    dataset: Path,
    rows: list[dict[str, str]],
    x: np.ndarray,
    y: np.ndarray,
    state: dict[str, Any],
) -> None:
    best = state.get("best_candidate")
    if not best:
        raise RuntimeError("cannot finalize without completed CV candidates")
    params = best["params"]
    train_index = np.asarray(state["train_index"], dtype=np.int64)
    test_index = np.asarray(state["test_index"], dtype=np.int64)

    train_scale = _feature_scale(x[train_index])
    holdout_model = _fit_model(x[train_index], y[train_index], train_scale, params)
    train_prediction = holdout_model.predict(_transform(x[train_index], train_scale))
    test_prediction = holdout_model.predict(_transform(x[test_index], train_scale))
    _atomic_joblib_dump(
        output_dir / "svr_model_holdout.joblib",
        _model_payload(holdout_model, train_scale, params, dataset, TARGET),
    )

    all_scale = _feature_scale(x)
    all_model = _fit_model(x, y, all_scale, params)
    _atomic_joblib_dump(
        output_dir / "svr_model_all_data.joblib",
        _model_payload(all_model, all_scale, params, dataset, TARGET),
    )
    _write_predictions(
        output_dir / "holdout_predictions.csv",
        rows,
        train_index,
        test_index,
        train_prediction,
        test_prediction,
    )
    _write_json(
        output_dir / "metrics.json",
        {
            "format_version": 1,
            "created_utc": _utc_now(),
            "dataset": str(dataset),
            "target": TARGET,
            "features": FEATURE_NAMES,
            "normalization": "only c, T and R use train/all-data maximum normalization; epsilon and target unchanged",
            "records_valid": len(rows),
            "train_records": int(train_index.size),
            "test_records": int(test_index.size),
            "train_groups": state["train_groups"],
            "test_groups": state["test_groups"],
            "best_params": params,
            "best_cv_rmse": best["mean_cv_rmse"],
            "best_cv_rmse_std": best["std_cv_rmse"],
            "train_metrics": _metrics(y[train_index], train_prediction),
            "test_metrics": _metrics(y[test_index], test_prediction),
        },
    )
    _write_json(
        output_dir / "train_manifest.json",
        {
            "format_version": 1,
            "created_utc": _utc_now(),
            "dataset": str(dataset),
            "dataset_sha256": state["dataset_sha256"],
            "target": TARGET,
            "features": FEATURE_NAMES,
            "split_group": "source_directory",
            "seed": state["seed"],
            "test_size": state["test_size"],
            "cv_splits": state["cv_splits"],
            "train_groups": state["train_groups"],
            "test_groups": state["test_groups"],
            "best_params": params,
            "J_definition": "average complete K=20 window costs, then reciprocal; Wx=I_10",
            "old_J_proxy_is_not_used": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")
    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be >= 2")

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, x, y, groups = _read_dataset(dataset)
    group_count = len(set(groups.tolist()))
    if group_count < 3:
        raise ValueError(f"need at least 3 source_directory groups, found {group_count}")

    checkpoint = output_dir / "checkpoint.json"
    scores_path = output_dir / "cv_scores.csv"
    state = _validate_or_load_state(
        checkpoint,
        dataset,
        _sha256(dataset),
        args.seed,
        args.test_size,
        args.cv_splits,
        x,
        y,
        groups,
    )
    if state.get("status") == "complete":
        print(f"Training already complete: {output_dir}")
        return 0

    candidates = _candidates()
    completed = state.setdefault("completed_candidates", {})
    train_index = np.asarray(state["train_index"], dtype=np.int64)
    train_x = x[train_index]
    train_y = y[train_index]
    train_groups = groups[train_index]
    try:
        for candidate_index, params in enumerate(candidates):
            key = _params_key(params)
            if key in completed:
                continue
            state["active_candidate_index"] = candidate_index
            state["status"] = "searching"
            _save_state(checkpoint, scores_path, state)
            result = _evaluate_candidate(
                train_x,
                train_y,
                train_groups,
                params,
                int(state["cv_splits"]),
            )
            result["candidate_index"] = candidate_index
            completed[key] = result
            state["active_candidate_index"] = None
            _save_state(checkpoint, scores_path, state)
            print(
                f"candidate={candidate_index + 1}/{len(candidates)} "
                f"params={params} mean_cv_rmse={result['mean_cv_rmse']:.9g}"
            )
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        _save_state(checkpoint, scores_path, state)
        print("Interrupted. Re-run the exact same command to resume.")
        return 130

    state["active_candidate_index"] = None
    state["status"] = "finalizing"
    _save_state(checkpoint, scores_path, state)
    try:
        _finalize(output_dir, dataset, rows, x, y, state)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        _save_state(checkpoint, scores_path, state)
        print("Interrupted during finalization. Re-run the exact same command to retry.")
        return 130

    state["status"] = "complete"
    state["completed_utc"] = _utc_now()
    _save_state(checkpoint, scores_path, state)
    print(f"Training complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

