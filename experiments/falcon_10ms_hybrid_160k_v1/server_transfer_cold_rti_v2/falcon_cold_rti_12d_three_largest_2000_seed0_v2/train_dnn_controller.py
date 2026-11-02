"""Train and evaluate the FP32 FALCON MPC-imitation controller.

The fixed environment split is respected: validation selects the checkpoint,
and the held-out test set is read only after training has finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from dnn_model_utils import build_model, load_normalization, model_spec, state_dict_cpu


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Use the requested training environment and "
            "install requirements-training.txt if necessary."
        ) from exc
    return torch


def _load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("X", "U")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    arrays["X"] = arrays["X"].astype(np.float32, copy=False)
    arrays["U"] = arrays["U"].astype(np.float32, copy=False)
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] < 1:
        raise ValueError(f"{path}: X must have shape (N, D), got {arrays['X'].shape}")
    if arrays["U"].ndim != 2 or arrays["U"].shape[1] != 4:
        raise ValueError(f"{path}: U must have shape (N, 4), got {arrays['U'].shape}")
    if arrays["X"].shape[0] != arrays["U"].shape[0]:
        raise ValueError(f"{path}: X/U sample counts differ")
    if not np.isfinite(arrays["X"]).all() or not np.isfinite(arrays["U"]).all():
        raise ValueError(f"{path}: X/U contain non-finite values")
    return arrays


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(torch: Any, requested: str) -> Any:
    selected = "cuda" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested
    )
    result = torch.device(selected)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    return result


def _normalized(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((array - mean) / std).astype(np.float32, copy=False)


def _make_loaders(torch: Any, arrays: dict[str, np.ndarray], normalization: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[Any, Any, Any]:
    x_mean, x_std = normalization["x_mean"], normalization["x_std"]
    u_mean, u_std = normalization["u_mean"], normalization["u_std"]
    dataset = torch.utils.data.TensorDataset
    loader = torch.utils.data.DataLoader
    train_set = dataset(
        torch.from_numpy(_normalized(arrays["train"]["X"], x_mean, x_std)),
        torch.from_numpy(_normalized(arrays["train"]["U"], u_mean, u_std)),
    )
    val_set = dataset(
        torch.from_numpy(_normalized(arrays["validation"]["X"], x_mean, x_std)),
        torch.from_numpy(_normalized(arrays["validation"]["U"], u_mean, u_std)),
    )
    test_set = dataset(
        torch.from_numpy(_normalized(arrays["test"]["X"], x_mean, x_std)),
        torch.from_numpy(_normalized(arrays["test"]["U"], u_mean, u_std)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    use_pin = (
        not bool(getattr(args, "disable_pin_memory", False))
        and (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
    )
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": use_pin}
    return (
        loader(train_set, shuffle=True, generator=generator, **common),
        loader(val_set, shuffle=False, **common),
        loader(test_set, shuffle=False, **common),
    )


def _run_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    gradient_clip_norm: float = 0.0,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), target)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_count = int(x.shape[0])
        total += float(loss.item()) * batch_count
        samples += batch_count
    return total / max(samples, 1)


def _predict(model: Any, loader: Any, device: Any, torch: Any) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _physical_metrics(predicted_norm: np.ndarray, target_norm: np.ndarray, normalization: dict[str, np.ndarray]) -> dict[str, Any]:
    prediction = predicted_norm * normalization["u_std"] + normalization["u_mean"]
    target = target_norm * normalization["u_std"] + normalization["u_mean"]
    error = prediction - target
    per_output_mse = np.mean(error ** 2, axis=0)
    per_output_mae = np.mean(np.abs(error), axis=0)
    names = ("collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z")
    return {
        "samples": int(target.shape[0]),
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mse_per_output": {name: float(value) for name, value in zip(names, per_output_mse)},
        "mae_per_output": {name: float(value) for name, value in zip(names, per_output_mae)},
    }


def _normalized_metrics(predicted_norm: np.ndarray, target_norm: np.ndarray) -> dict[str, Any]:
    """Return dimensionless metrics in the exact space optimized by training."""

    error = np.asarray(predicted_norm) - np.asarray(target_norm)
    per_output_mse = np.mean(error ** 2, axis=0)
    per_output_mae = np.mean(np.abs(error), axis=0)
    names = ("collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z")
    return {
        "samples": int(target_norm.shape[0]),
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mse_per_output": {name: float(value) for name, value in zip(names, per_output_mse)},
        "mae_per_output": {name: float(value) for name, value in zip(names, per_output_mae)},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_normalized_mse",
                "validation_normalized_mse",
                "validation_physical_mse",
                "learning_rate",
            ),
        )
        writer.writeheader()
        writer.writerows(history)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                "epoch": int(row["epoch"]),
                "train_normalized_mse": float(row["train_normalized_mse"]),
                "validation_normalized_mse": (
                    float(row["validation_normalized_mse"])
                    if row.get("validation_normalized_mse") not in (None, "")
                    else float("nan")
                ),
                "validation_physical_mse": float(row["validation_physical_mse"]),
                "learning_rate": (
                    float(row["learning_rate"])
                    if row.get("learning_rate") not in (None, "")
                    else float("nan")
                ),
            }
            for row in csv.DictReader(handle)
        ]


def _per_gate_metrics(predicted_norm: np.ndarray, target_norm: np.ndarray, split: dict[str, np.ndarray], normalization: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    prediction = predicted_norm * normalization["u_std"] + normalization["u_mean"]
    target = target_norm * normalization["u_std"] + normalization["u_mean"]
    error = prediction - target
    if "target_gate_id" not in split:
        return []
    rows = []
    for gate in sorted(np.unique(split["target_gate_id"]).astype(int).tolist()):
        mask = split["target_gate_id"] == gate
        rows.append({
            "target_gate_id": gate,
            "samples": int(mask.sum()),
            "mse": float(np.mean(error[mask] ** 2)),
            "mae": float(np.mean(np.abs(error[mask]))),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
        })
    return rows


def _save_curve(path: Path, history: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping training curve because matplotlib could not be loaded: {exc}", flush=True)
        return
    epochs = [row["epoch"] for row in history]
    train = [row["train_normalized_mse"] for row in history]
    validation = [
        row.get("validation_normalized_mse", row["validation_physical_mse"])
        for row in history
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, train, label="train normalized MSE")
    axis.plot(epochs, validation, label="validation normalized MSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _baseline_metrics(
    target_norm: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Measure the physical error of always predicting the training-set mean."""
    return _physical_metrics(np.zeros_like(target_norm), target_norm, normalization)


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    if args.patience < 0 or args.min_epochs < 0:
        raise ValueError("patience and min-epochs must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early-stopping-min-delta must be non-negative")
    if args.gradient_clip_norm < 0.0:
        raise ValueError("gradient-clip-norm must be non-negative")
    if args.lr_plateau_patience < 0 or args.lr_plateau_cooldown < 0:
        raise ValueError("LR plateau patience and cooldown must be non-negative")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        raise ValueError("lr-plateau-factor must be strictly between 0 and 1")
    if args.lr_plateau_threshold < 0.0:
        raise ValueError("lr-plateau-threshold must be non-negative")
    if args.min_learning_rate < 0.0 or args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate must be between 0 and learning-rate")
    _seed_everything(torch, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: _load_split(args.data_dir / f"{name}.npz") for name in ("train", "validation", "test")}
    input_dims = {int(value["X"].shape[1]) for value in arrays.values()}
    if len(input_dims) != 1:
        raise ValueError(f"dataset splits use inconsistent input dimensions: {sorted(input_dims)}")
    input_dim = input_dims.pop()
    split_manifest_path = args.data_dir / "split_manifest.json"
    split_manifest: dict[str, Any] = {}
    if split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if int(split_manifest.get("input_dimensions", input_dim)) != input_dim:
            raise ValueError("split manifest input dimension does not match split arrays")
    input_features = split_manifest.get("input_features")
    teacher_policy = split_manifest.get("teacher_policy")
    input_contract = split_manifest.get("input_contract")
    reference_speed_values = split_manifest.get("reference_speed_values", [])
    normalization = load_normalization(
        args.data_dir / "normalization.json",
        expected_input_dim=input_dim,
    )
    train_loader, val_loader, test_loader = _make_loaders(torch, arrays, normalization, args)
    device = _device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(torch, depth=args.depth, width=args.width, input_dim=input_dim).to(device)
    # The foreach/multi-tensor Adam path is faster for many tiny parameter
    # tensors, but on Windows WDDM it has repeatedly surfaced a lost CUDA
    # context as ``CUDA error: unknown error`` after interrupted/resumed long
    # runs.  These FC models have few large tensors, so the conservative
    # single-tensor path has negligible throughput cost and is substantially
    # easier on the display-driver watchdog.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, foreach=False
    )
    scheduler = None
    if args.lr_plateau_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            threshold=args.lr_plateau_threshold,
            threshold_mode="rel",
            cooldown=args.lr_plateau_cooldown,
            min_lr=args.min_learning_rate,
        )
    criterion = torch.nn.MSELoss()
    history_path = args.output_dir / "history.csv"
    history = _load_history(history_path)
    start_epoch = 1
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stop_reference = float("inf")
    resume_path = args.resume_from or (args.output_dir / "latest_checkpoint.pt" if args.resume else None)
    if resume_path is not None and args.warm_start_from is not None:
        raise ValueError("resume and warm-start options are mutually exclusive")
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_selection_metric = str(
            checkpoint.get("selection_metric", "physical_mse")
        )
        if checkpoint_selection_metric != args.selection_metric:
            raise ValueError(
                "resume checkpoint selection metric does not match this run: "
                f"{checkpoint_selection_metric!r} != {args.selection_metric!r}"
            )
        checkpoint_input_dim = int(checkpoint.get("input_dim", 12))
        if (
            int(checkpoint["depth"]) != args.depth
            or int(checkpoint["width"]) != args.width
            or checkpoint_input_dim != input_dim
        ):
            raise ValueError("resume checkpoint architecture does not match the requested dataset/model")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for parameter_group in optimizer.param_groups:
            parameter_group["foreach"] = False
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        if "early_stop_reference" in checkpoint:
            early_stop_reference = float(checkpoint["early_stop_reference"])
            stale_epochs = int(checkpoint["stale_epochs"])
        else:
            # Old checkpoints counted every arbitrarily small improvement as
            # progress.  Start the new meaningful-improvement clock afresh so
            # enabling early stopping cannot terminate immediately on resume.
            early_stop_reference = best_validation
            stale_epochs = 0
        if not history or history[-1]["epoch"] < int(checkpoint["epoch"]):
            history = list(checkpoint.get("history", history))
        print(
            f"Resumed from {resume_path} at epoch {start_epoch}; "
            f"best_epoch={best_epoch} best_validation_mse={best_validation:.8f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
    elif args.warm_start_from is not None:
        if not args.warm_start_from.exists():
            raise FileNotFoundError(f"warm-start checkpoint does not exist: {args.warm_start_from}")
        checkpoint = torch.load(args.warm_start_from, map_location=device, weights_only=False)
        checkpoint_input_dim = int(checkpoint.get("input_dim", 12))
        if (
            int(checkpoint["depth"]) != args.depth
            or int(checkpoint["width"]) != args.width
            or checkpoint_input_dim != input_dim
        ):
            raise ValueError("warm-start checkpoint architecture does not match the requested dataset/model")
        model.load_state_dict(checkpoint["model_state"])
        print(
            f"Warm-started model weights from {args.warm_start_from}; "
            "Adam optimizer, epoch counter, and early-stopping state were reset.",
            flush=True,
        )

    started = time.perf_counter()
    epochs_completed = start_epoch - 1
    stopped_early = bool(args.finalize_best)
    termination_reason = "manual_finalize_best" if args.finalize_best else "maximum_epochs"
    if args.finalize_best:
        best_path = args.output_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"cannot finalize because best_model.pt is missing: {best_path}"
            )
        if history:
            epochs_completed = max(int(row["epoch"]) for row in history)
        print(
            f"Finalizing the existing best checkpoint without additional training: "
            f"epochs_completed={epochs_completed} best_epoch={best_epoch} "
            f"best_validation_mse={best_validation:.8f}",
            flush=True,
        )
        epoch_range = range(0)
    else:
        epoch_range = range(start_epoch, args.epochs + 1)
    for epoch in epoch_range:
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            torch,
            gradient_clip_norm=float(args.gradient_clip_norm),
        )
        validation_norm = _predict(model, val_loader, device, torch)
        validation_target = np.concatenate([batch[1].numpy() for batch in val_loader], axis=0)
        validation_physical = _physical_metrics(validation_norm, validation_target, normalization)
        validation_normalized = _normalized_metrics(validation_norm, validation_target)
        validation_mse = float(
            validation_normalized["mse"]
            if args.selection_metric == "normalized_mse"
            else validation_physical["mse"]
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_normalized["mse"],
            "validation_physical_mse": validation_physical["mse"],
            "learning_rate": learning_rate,
        }
        history = [existing for existing in history if existing["epoch"] != epoch]
        history.append(row)
        history.sort(key=lambda item: item["epoch"])
        epochs_completed = epoch
        improved = validation_mse < best_validation
        meaningful_improvement = validation_mse < (
            early_stop_reference - args.early_stopping_min_delta
        )
        if meaningful_improvement:
            early_stop_reference = validation_mse
            stale_epochs = 0
        else:
            stale_epochs += 1
        if improved:
            best_validation = validation_mse
            best_epoch = epoch
            best_payload = {
                "format_version": 2,
                "epoch": epoch,
                "depth": args.depth,
                "width": args.width,
                "input_dim": input_dim,
                "input_features": input_features,
                "teacher_policy": teacher_policy,
                "input_contract": input_contract,
                "reference_speed_values": reference_speed_values,
                "initialization": "kaiming_relu_hidden_xavier_output_v1",
                "selection_metric": args.selection_metric,
                "gradient_clip_norm": float(args.gradient_clip_norm),
                "model_state": state_dict_cpu(model),
                "normalization": {key: value.tolist() for key, value in normalization.items()},
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "early_stop_reference": early_stop_reference,
                "history": history,
            }
            _atomic_torch_save(torch, best_payload, args.output_dir / "best_model.pt")
        previous_learning_rate = learning_rate
        if scheduler is not None:
            scheduler.step(validation_mse)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        if current_learning_rate < previous_learning_rate:
            print(
                f"Learning rate reduced at epoch {epoch}: "
                f"{previous_learning_rate:.3e} -> {current_learning_rate:.3e}",
                flush=True,
            )
        latest_payload = {
            "format_version": 2,
            "epoch": epoch,
            "depth": args.depth,
            "width": args.width,
            "input_dim": input_dim,
            "input_features": input_features,
            "teacher_policy": teacher_policy,
            "input_contract": input_contract,
            "reference_speed_values": reference_speed_values,
            "initialization": "kaiming_relu_hidden_xavier_output_v1",
            "selection_metric": args.selection_metric,
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "seed": args.seed,
            "model_state": state_dict_cpu(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "normalization": {key: value.tolist() for key, value in normalization.items()},
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "early_stop_reference": early_stop_reference,
            "history": history,
        }
        _atomic_torch_save(torch, latest_payload, args.output_dir / "latest_checkpoint.pt")
        _save_history(history_path, history)
        if epoch == start_epoch or epoch % args.log_every == 0 or improved:
            print(
                f"epoch={epoch:05d}/{args.epochs} train_normalized_mse={train_loss:.8f} "
                f"validation_normalized_mse={validation_normalized['mse']:.8f} "
                f"validation_physical_mse={validation_physical['mse']:.8f} "
                f"selection={args.selection_metric} best={best_validation:.8f} "
                f"stale={stale_epochs} lr={current_learning_rate:.3e}", flush=True
            )
        if (
            args.patience > 0
            and epoch >= args.min_epochs
            and stale_epochs >= args.patience
            and (
                not args.early_stop_after_min_lr
                or scheduler is None
                or current_learning_rate
                <= args.min_learning_rate * (1.0 + 1e-12)
            )
        ):
            stopped_early = True
            termination_reason = "early_stopping"
            print(
                f"Early stopping after {epoch} epochs: no validation improvement "
                f"larger than {args.early_stopping_min_delta:g} for "
                f"{stale_epochs} epochs; test remains untouched until final evaluation.",
                flush=True,
            )
            break

    best_path = args.output_dir / "best_model.pt"
    if not best_path.exists():
        raise RuntimeError("training ended without producing best_model.pt")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_norm = _predict(model, test_loader, device, torch)
    test_target = np.concatenate([batch[1].numpy() for batch in test_loader], axis=0)
    validation_split = arrays["validation"]
    test_split = arrays["test"]
    validation_norm = _predict(model, val_loader, device, torch)
    validation_target = np.concatenate([batch[1].numpy() for batch in val_loader], axis=0)
    validation_metrics = _physical_metrics(validation_norm, validation_target, normalization)
    test_metrics = _physical_metrics(test_norm, test_target, normalization)
    validation_normalized_metrics = _normalized_metrics(validation_norm, validation_target)
    test_normalized_metrics = _normalized_metrics(test_norm, test_target)
    validation_baseline = _baseline_metrics(validation_target, normalization)
    baseline_mse = float(validation_baseline["mse"])
    validation_baseline_ratio = (
        float(validation_metrics["mse"]) / baseline_mse
        if baseline_mse > 0.0
        else float("inf")
    )
    health = {
        "validation_mean_baseline_mse": baseline_mse,
        "validation_model_to_baseline_mse_ratio": validation_baseline_ratio,
        "maximum_allowed_ratio": args.max_validation_baseline_ratio,
        "passed": (
            args.max_validation_baseline_ratio is None
            or validation_baseline_ratio <= args.max_validation_baseline_ratio
        ),
    }
    if not health["passed"]:
        _write_json(
            args.output_dir / "health_failure.json",
            {
                "status": "rejected",
                "reason": "model did not improve sufficiently over the mean-output baseline",
                "health": health,
                "best_epoch": best_epoch,
                "epochs_completed": epochs_completed,
            },
        )
        raise RuntimeError(
            "model health check failed: validation MSE / mean-baseline MSE = "
            f"{validation_baseline_ratio:.6f}, required <= "
            f"{args.max_validation_baseline_ratio:.6f}"
        )
    test_prediction_physical = test_norm * normalization["u_std"] + normalization["u_mean"]
    test_target_physical = test_target * normalization["u_std"] + normalization["u_mean"]
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        U_pred=test_prediction_physical.astype(np.float32),
        U_true=test_target_physical.astype(np.float32),
        error=(test_prediction_physical - test_target_physical).astype(np.float32),
        environment_id=test_split.get("environment_id", np.empty(0, dtype=np.int64)),
        episode_id=test_split.get("episode_id", np.empty(0, dtype=np.int64)),
        target_gate_id=test_split.get("target_gate_id", np.empty(0, dtype=np.int64)),
        data_source=test_split.get("data_source", np.empty(0, dtype=np.int8)),
        sample_family=test_split.get("sample_family", np.empty(0, dtype=np.int8)),
    )
    output_names = ("collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z")
    with (args.output_dir / "per_output_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("output", "mse", "mae"))
        for name in output_names:
            writer.writerow((name, test_metrics["mse_per_output"][name], test_metrics["mae_per_output"][name]))
    with (args.output_dir / "per_gate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("target_gate_id", "samples", "mse", "mae", "rmse"))
        writer.writeheader()
        writer.writerows(_per_gate_metrics(test_norm, test_target, test_split, normalization))
    model_cpu = model.cpu().eval()
    torch.jit.script(model_cpu).save(str(args.output_dir / "controller_scripted.pt"))
    spec = model_spec(
        depth=args.depth,
        width=args.width,
        input_dim=input_dim,
        input_features=input_features,
        teacher_policy=teacher_policy,
        input_contract=input_contract,
        normalization_path=str((args.data_dir / "normalization.json").resolve()),
    )
    spec["checkpoint"] = str(best_path.resolve())
    spec["reference_speed_values"] = reference_speed_values
    _write_json(args.output_dir / "model_spec.json", spec)
    _write_json(args.output_dir / "normalization.json", {key: value.tolist() for key, value in normalization.items()})
    _save_history(history_path, history)
    if not args.skip_curve:
        _save_curve(args.output_dir / "training_curve.png", history)
    metrics = {
        "format_version": 2,
        "depth": args.depth,
        "width": args.width,
        "input_dimensions": input_dim,
        "input_features": spec["input_features"],
        "teacher_policy": spec["teacher_policy"],
        "input_contract": spec.get("input_contract"),
        "reference_speed_values": reference_speed_values,
        "initialization": "kaiming_relu_hidden_xavier_output_v1",
        "selection_metric": args.selection_metric,
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "pin_memory": not bool(args.disable_pin_memory),
        "metric_definition": {
            "physical_mse": "mean((U_pred_physical-U_true_physical)^2) over all samples and four outputs",
            "normalized_mse": "mean((U_pred_normalized-U_true_normalized)^2) over all samples and four outputs",
            "training_objective": "normalized_mse",
        },
        "parameters": int(sum(parameter.numel() for parameter in model_cpu.parameters())),
        "device": str(device),
        "seed": args.seed,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "training_complete": True,
        "termination_reason": termination_reason,
        "stopped_early": stopped_early,
        "early_stopping": {
            "enabled": args.patience > 0,
            "patience": args.patience,
            "minimum_epochs": args.min_epochs,
            "minimum_absolute_improvement": args.early_stopping_min_delta,
            "requires_minimum_learning_rate": args.early_stop_after_min_lr,
        },
        "learning_rate_schedule": {
            "type": "reduce_on_plateau" if scheduler is not None else "constant",
            "initial_learning_rate": args.learning_rate,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "plateau_patience": args.lr_plateau_patience,
            "plateau_factor": args.lr_plateau_factor,
            "plateau_relative_threshold": args.lr_plateau_threshold,
            "plateau_cooldown": args.lr_plateau_cooldown,
            "minimum_learning_rate": args.min_learning_rate,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_normalized": validation_normalized_metrics,
        "test_normalized": test_normalized_metrics,
        "health": health,
        "elapsed_seconds": time.perf_counter() - started,
        "test_used_for_early_stopping": False,
        "data_dir": str(args.data_dir.resolve()),
    }
    if device.type == "cuda":
        metrics["cuda_memory"] = {
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
        }
    _write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dnn_dataset_full_route_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("dnn_runs/depth10_width512"))
    parser.add_argument("--depth", type=int, choices=(10, 22, 34), default=10)
    parser.add_argument("--width", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--epochs", type=int, default=20000, help="total target epochs, including resumed epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
        help="clip the global gradient norm to this value; 0 disables clipping",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="do not early-stop before this epoch",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="minimum absolute validation-MSE decrease that resets early stopping",
    )
    parser.add_argument(
        "--early-stop-after-min-lr",
        action="store_true",
        help="when LR scheduling is enabled, permit early stopping only after reaching min LR",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=0,
        help="epochs without improvement before reducing LR; 0 disables the scheduler",
    )
    parser.add_argument("--lr-plateau-factor", type=float, default=0.3)
    parser.add_argument("--lr-plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--selection-metric",
        choices=("physical_mse", "normalized_mse"),
        default="physical_mse",
        help=(
            "validation metric used by best-checkpoint selection, LR scheduling, "
            "and early stopping; normalized_mse matches the training objective"
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="use ordinary host memory for DataLoader batches (safer after Windows CUDA TDRs)",
    )
    parser.add_argument("--resume", action="store_true", help="resume from output-dir/latest_checkpoint.pt")
    parser.add_argument(
        "--finalize-best",
        action="store_true",
        help=(
            "skip additional epochs and produce final metrics/exports from the existing "
            "best_model.pt; normally used together with --resume"
        ),
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--warm-start-from",
        type=Path,
        default=None,
        help="load only model weights from a checkpoint and reset optimizer/training state",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device")
    parser.add_argument(
        "--max-validation-baseline-ratio",
        type=float,
        default=None,
        help=(
            "optional post-training health gate: validation physical MSE must be at most "
            "this fraction of the error from always predicting the training-set mean"
        ),
    )
    parser.add_argument(
        "--skip-curve",
        action="store_true",
        help="do not import matplotlib or save training_curve.png",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
