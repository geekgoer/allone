"""Train one MLP on a frozen 12D split or a separate decision experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from .compute_complexity import checkpoint_size, complexity
from .eval_offline import evaluate, load_normalization, load_split
from .models.factory import build_model


ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("hidden_layers", "hidden_width", "seed", "learning_rate", "batch_size", "max_epochs", "patience", "min_epochs", "gradient_clip_norm", "data_dir")


def _curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    values = [float(row["validation_normalized_mse"]) for row in rows]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-12)
    last_epoch = max(int(rows[-1]["epoch"]), 2)
    points = " ".join(f"{40 + 720 * (int(row['epoch']) - 1) / (last_epoch - 1):.1f},{340 - 300 * (float(row['validation_normalized_mse']) - lo) / span:.1f}" for row in rows)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="380"><rect width="800" height="380" fill="white"/><text x="40" y="25">Validation normalized MSE (lower is better)</text><path d="M40 40 V340 H760" fill="none" stroke="black"/><polyline points="{points}" fill="none" stroke="#1267ad" stroke-width="2"/><text x="40" y="365">Epoch 1 to {last_epoch}; min={lo:.6g}</text></svg>\n', encoding="utf-8")


def _save_checkpoint(path: Path, torch, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def learning_rate_for_epoch(config: dict, epoch: int) -> float:
    base = float(config["learning_rate"])
    if config.get("lr_schedule", "constant") == "constant":
        return base
    minimum = float(config["min_learning_rate"])
    progress = (epoch - 1) / max(int(config["max_epochs"]) - 1, 1)
    return minimum + (base - minimum) * (1 + math.cos(math.pi * progress)) / 2


def train(config: dict, output: Path, *, resume: bool = False, device: str = "auto") -> dict:
    import torch

    if any(key not in config for key in REQUIRED):
        raise ValueError(f"missing config fields: {sorted(set(REQUIRED) - set(config))}")
    model_type = config.get("model_type")
    if model_type not in ("plain_mlp", "residual_mlp") or config.get("activation") != "relu" or config.get("residual") is not (model_type == "residual_mlp"):
        raise ValueError("model_type, residual and ReLU configuration are inconsistent")
    input_dim = int(config.get("input_dim", 0))
    if input_dim not in (12, 237, 300) or config.get("output_dim") != 4:
        raise ValueError("requires input_dim=12, 237 or 300, output_dim=4")
    if min(int(config[key]) for key in ("hidden_layers", "hidden_width", "batch_size", "max_epochs", "patience", "min_epochs")) < 1:
        raise ValueError("model and training counts must be positive")
    if int(config["min_epochs"]) > int(config["max_epochs"]):
        raise ValueError("min_epochs exceeds max_epochs")
    schedule = config.get("lr_schedule", "constant")
    weight_decay = float(config.get("weight_decay", 0.))
    if schedule not in ("constant", "cosine") or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("requires constant/cosine schedule and nonnegative weight_decay")
    if schedule == "cosine" and not 0 < float(config.get("min_learning_rate", 0.)) <= float(config["learning_rate"]):
        raise ValueError("cosine requires 0 < min_learning_rate <= learning_rate")
    log_every = int(config.get("log_every", 10))
    if log_every < 1:
        raise ValueError("log_every must be positive")
    checkpoint_every = int(config.get("checkpoint_every", 1))
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    early_stopping = config.get("early_stopping", True)
    restore_initial_optimizer = config.get("restore_initial_optimizer", False)
    if not isinstance(early_stopping, bool) or not isinstance(restore_initial_optimizer, bool):
        raise ValueError("early_stopping and restore_initial_optimizer must be booleans")
    if restore_initial_optimizer and not config.get("init_checkpoint"):
        raise ValueError("restore_initial_optimizer requires init_checkpoint")
    preserve_initial = bool(config.get("preserve_initial_best", False))
    if preserve_initial and not config.get("init_checkpoint"):
        raise ValueError("preserve_initial_best requires init_checkpoint")
    evaluation_splits = config.get("evaluation_splits", ["validation", "test"])
    if "validation" not in evaluation_splits or not set(evaluation_splits) <= {"train", "validation", "test"}:
        raise ValueError("evaluation_splits must include validation and contain valid splits")
    if config.get("init_checkpoint") and output.resolve() == (ROOT / config["init_checkpoint"]).resolve().parent:
        raise ValueError("warm-start output must not overwrite the source run")
    data_dir = ROOT / config["data_dir"]
    norm = load_normalization(data_dir)
    if len(norm["x_mean"]) != input_dim:
        raise ValueError("config and normalization input dimensions disagree")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        if not resume or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(f"run exists with a different config or without --resume: {output}")
    else:
        if resume:
            raise FileNotFoundError(f"no saved run to resume: {output}")
        _save_json(config_path, config)
    last_path = output / "last_model.pt"
    if resume and not last_path.exists():
        if input_dim in (237, 300) and not (output / "train_log.csv").exists():
            print("Pilot interrupted before its first epoch; restarting from the saved seed", flush=True)
        elif not preserve_initial or (output / "train_log.csv").exists():
            raise FileNotFoundError(f"cannot resume without a saved epoch: {last_path}")
        else:
            print("Initialization was interrupted; restarting from the source checkpoint", flush=True)
        resume = False
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(torch, model_type=model_type, input_dim=input_dim, output_dim=4,
                        hidden_layers=int(config["hidden_layers"]), hidden_width=int(config["hidden_width"])).to(selected_device)
    if not resume and config.get("init_checkpoint"):
        init_path = ROOT / config["init_checkpoint"]
        source = torch.load(init_path, map_location="cpu", weights_only=False)
        if int(source.get("input_dim", 12)) != input_dim:
            raise ValueError("initial checkpoint input dimensions disagree")
        if (int(source["hidden_layers"]), int(source["hidden_width"]), source.get("model_type", "plain_mlp")) != (
            int(config["hidden_layers"]), int(config["hidden_width"]), model_type
        ):
            raise ValueError(f"initial checkpoint architecture mismatch: {init_path}")
        model.load_state_dict(source["model_state"], strict=True)
        print(f"Warm-started from {init_path}; optimizer state {'restored' if restore_initial_optimizer else 'reset'}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=weight_decay, foreach=False)
    if restore_initial_optimizer and not resume:
        optimizer.load_state_dict(source["optimizer_state"])
        for group in optimizer.param_groups:
            group.update(lr=float(config["learning_rate"]), weight_decay=weight_decay)
    start_epoch, best, stale = 1, float("inf"), 0
    history = []
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_epoch, best, stale = int(saved["epoch"]) + 1, float(saved["best_validation_normalized_mse"]), int(saved["stale"])
        history = saved["history"]
        torch.set_rng_state(saved["torch_rng_state"])
        if selected_device.startswith("cuda") and "cuda_rng_states" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        # A shutdown can occur after saving last but before saving the new best.
        if int(saved["epoch"]) == 0 or (history and history[-1]["validation_normalized_mse"] == best):
            _save_checkpoint(output / "best_model.pt", torch, saved)

    def checkpoint_payload(epoch: int) -> dict:
        payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                   "best_validation_normalized_mse": best, "stale": stale,
                   "hidden_layers": int(config["hidden_layers"]), "hidden_width": int(config["hidden_width"]),
                   "model_type": model_type, "input_dim": input_dim,
                   "history": history, "torch_rng_state": torch.get_rng_state()}
        if selected_device.startswith("cuda"):
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        return payload

    if preserve_initial and not resume:
        initial = {split: evaluate(model, data_dir, split, device=selected_device)
                   for split in ("train", "validation")}
        initial["source_checkpoint"] = str((ROOT / config["init_checkpoint"]).resolve())
        initial["source_epoch"] = int(source["epoch"])
        best = initial["validation"]["normalized_mse"]
        _save_json(output / "initial_metrics.json", initial)
        payload = checkpoint_payload(0)
        _save_checkpoint(output / "best_model.pt", torch, payload)
        _save_checkpoint(last_path, torch, payload)
        print(f"Initial checkpoint retained: validation_normalized_mse={best:.8g} (epoch=0)", flush=True)
    if input_dim in (237, 300) and not resume and not preserve_initial:
        _save_checkpoint(last_path, torch, checkpoint_payload(0))
    x_train, u_train = load_split(data_dir, "train", config.get("train_limit"))
    x_val, u_val = load_split(data_dir, "validation", config.get("validation_limit"))
    if x_train.shape[1] != input_dim or x_val.shape[1] != input_dim:
        raise ValueError("split and config input dimensions disagree")
    x_train = torch.from_numpy((x_train - norm["x_mean"]) / norm["x_std"])
    u_train = torch.from_numpy((u_train - norm["u_mean"]) / norm["u_std"])
    x_val = torch.from_numpy((x_val - norm["x_mean"]) / norm["x_std"])
    u_val = torch.from_numpy((u_val - norm["u_mean"]) / norm["u_std"])
    log_path = output / "train_log.csv"
    fields = ("epoch", "train_normalized_mse", "validation_normalized_mse", "best_validation_normalized_mse", "stale", "mean_grad_norm_before_clip", "clip_fraction")
    if schedule != "constant":
        fields += ("learning_rate",)
    if resume:
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
    already_stopped = early_stopping and start_epoch - 1 >= int(config["min_epochs"]) and stale >= int(config["patience"])
    epochs = () if already_stopped else range(start_epoch, int(config["max_epochs"]) + 1)
    for epoch in epochs:
        # Epoch-derived schedule resumes without restarting a scheduler clock.
        if schedule != "constant":
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_for_epoch(config, epoch)
        model.train()
        order = torch.randperm(len(x_train))
        total_loss = 0.0
        grad_total, clipped, batch_count = 0.0, 0, 0
        for indices in order.split(int(config["batch_size"])):
            x = x_train[indices].to(selected_device)
            u = u_train[indices].to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - u) ** 2)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
            grad_total += grad_norm
            clipped += grad_norm > float(config["gradient_clip_norm"])
            batch_count += 1
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(x_val), 4096):
                x = x_val[start:start + 4096].to(selected_device)
                u = u_val[start:start + 4096].to(selected_device)
                val_sum += float(torch.sum((model(x) - u) ** 2).item())
        val_mse = val_sum / (len(x_val) * 4)
        improved = val_mse < best
        best, stale = (val_mse, 0) if improved else (best, stale + 1)
        row = {"epoch": epoch, "train_normalized_mse": total_loss / len(x_train), "validation_normalized_mse": val_mse,
               "best_validation_normalized_mse": best, "stale": stale,
               "mean_grad_norm_before_clip": grad_total / batch_count, "clip_fraction": clipped / batch_count}
        if schedule != "constant":
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        payload = checkpoint_payload(epoch)
        stopping = early_stopping and epoch >= int(config["min_epochs"]) and stale >= int(config["patience"])
        if improved or stopping or epoch % checkpoint_every == 0 or epoch == int(config["max_epochs"]):
            _save_checkpoint(last_path, torch, payload)
        if improved:
            _save_checkpoint(output / "best_model.pt", torch, payload)
        with log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        if epoch == 1 or epoch % log_every == 0 or improved:
            print(f"{output.name} epoch={epoch} train={row['train_normalized_mse']:.8g} val={val_mse:.8g} best={best:.8g} stale={stale}", flush=True)
        if stopping:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break
    _curve(output / "validation_curve.svg", history)
    best_model = build_model(torch, model_type=model_type, input_dim=input_dim, output_dim=4,
                             hidden_layers=int(config["hidden_layers"]), hidden_width=int(config["hidden_width"])).to(selected_device)
    best_saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    best_model.load_state_dict(best_saved["model_state"])
    metrics = {split: evaluate(best_model, data_dir, split, device=selected_device) for split in evaluation_splits}
    _save_json(output / "offline_metrics.json", metrics)
    cost = complexity(input_dim, 4, int(config["hidden_layers"]), int(config["hidden_width"]))
    cost["elementwise_residual_adds"] = (int(config["hidden_layers"]) - 1) * int(config["hidden_width"]) if model_type == "residual_mlp" else 0
    cost["elementwise_residual_multiplies"] = cost["elementwise_residual_adds"]
    cost["checkpoint_size_bytes"] = checkpoint_size(output / "best_model.pt")
    _save_json(output / "complexity.json", cost)
    _save_json(output / "completion.json", {
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_saved["epoch"]),
        "best_validation_normalized_mse": float(best_saved["best_validation_normalized_mse"]),
        "evaluation_splits": list(evaluation_splits),
        "test_used_for_selection": False,
        "initial_checkpoint_retained_as_candidate": preserve_initial,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metrics = train(config, args.output_dir, resume=args.resume, device=args.device)
    print(json.dumps(metrics.get("test", metrics["validation"]), indent=2))


if __name__ == "__main__":
    main()
