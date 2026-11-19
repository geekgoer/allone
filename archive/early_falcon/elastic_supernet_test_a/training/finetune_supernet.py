"""Sandwich-style fine-tuning for a shared-weight width-elastic MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from elastic_supernet_test_a.models.baseline_mlp import load_fixed_controller
from elastic_supernet_test_a.models.elastic_mlp import ElasticMLP, SubnetConfig


def _normalized_split(path: Path, fixed: Any, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        u = np.asarray(archive["U"], dtype=np.float32)
    if limit is not None:
        x, u = x[:limit], u[:limit]
    return (
        (x - fixed.x_mean.cpu().numpy()) / fixed.x_std.cpu().numpy(),
        (u - fixed.u_mean.cpu().numpy()) / fixed.u_std.cpu().numpy(),
    )


def _validation_score(
    torch: Any,
    elastic: ElasticMLP,
    fixed: Any,
    x: Any,
    u: Any,
    widths: list[int],
    batch_size: int,
    retention_weight: float,
) -> dict[str, float]:
    elastic.eval()
    scores: list[float] = []
    full_drift_total = 0.0
    full_drift_values = 0
    with torch.inference_mode():
        for width in widths:
            config = SubnetConfig.uniform(width, elastic.hidden_layers)
            total = 0.0
            values = 0
            for start in range(0, len(x), batch_size):
                prediction = elastic(x[start:start + batch_size], config)
                difference = prediction - u[start:start + batch_size]
                total += float(torch.square(difference).sum().item())
                values += difference.numel()
            scores.append(total / values)
        maximum = elastic.max_config
        for start in range(0, len(x), batch_size):
            batch_x = x[start:start + batch_size]
            elastic_full = elastic(batch_x, maximum)
            fixed_full = fixed.normalized_forward(batch_x)
            difference = elastic_full - fixed_full
            full_drift_total += float(torch.square(difference).sum().item())
            full_drift_values += difference.numel()
    family_mse = float(np.mean(scores))
    full_drift_mse = full_drift_total / full_drift_values
    return {
        "family_mse": family_mse,
        "full_vs_fixed_mse": full_drift_mse,
        "selection_score": family_mse + retention_weight * full_drift_mse,
    }


def _save_checkpoint(
    torch: Any,
    path: Path,
    elastic: ElasticMLP,
    fixed: Any,
    *,
    widths: list[int],
    validation_widths: list[int],
    retention_weight: float,
    epoch: int,
    score: float,
    validation: dict[str, float],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "elastic_state_dict": {
                key: value.detach().cpu() for key, value in elastic.state_dict().items()
            },
            "baseline": fixed.info.__dict__,
            "candidate_widths": widths,
            "validation_widths": validation_widths,
            "selection_metric": (
                "mean normalized task MSE over validation_widths + "
                f"{retention_weight} * maximum-subnet-vs-fixed normalized MSE"
            ),
            "best_epoch": epoch,
            "best_selection_score": score,
            "validation_at_best": validation,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("elastic_supernet_test_a/checkpoints/elastic_supernet_best.pt"))
    parser.add_argument("--widths", default="16,32,48,64,96,128,160,192,256,320,384,448,512")
    parser.add_argument("--validation-widths", default="16,64,128,256,384,512")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--random-subnets", type=int, default=2)
    parser.add_argument("--kd-weight", type=float, default=0.2)
    parser.add_argument("--full-loss-weight", type=float, default=4.0)
    parser.add_argument("--full-kd-weight", type=float, default=2.0)
    parser.add_argument("--selection-retention-weight", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = load_fixed_controller(torch, args.model, args.normalization, device=device)
    fixed.model.requires_grad_(False)
    elastic = ElasticMLP.from_fixed_controller(fixed).to(device)
    widths = sorted(set(int(value) for value in args.widths.split(",")))
    validation_widths = sorted(set(int(value) for value in args.validation_widths.split(",")))
    if widths[-1] != elastic.max_width or any(value not in widths for value in validation_widths):
        raise ValueError("width sets must include the maximum and validation widths must be candidates")

    train_x, train_u = _normalized_split(args.train, fixed, args.max_train_samples)
    val_x, val_u = _normalized_split(args.validation, fixed, args.max_validation_samples)
    train_x_t = torch.from_numpy(train_x)
    train_u_t = torch.from_numpy(train_u)
    val_x_t = torch.from_numpy(val_x).to(device)
    val_u_t = torch.from_numpy(val_u).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x_t, train_u_t),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    optimizer = torch.optim.Adam(elastic.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.MSELoss()
    initial_validation = _validation_score(
        torch,
        elastic,
        fixed,
        val_x_t,
        val_u_t,
        validation_widths,
        args.batch_size,
        args.selection_retention_weight,
    )
    best = initial_validation["selection_score"]
    stale = 0
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "train_loss": None,
        "validation_family_mse": initial_validation["family_mse"],
        "validation_full_vs_fixed_mse": initial_validation["full_vs_fixed_mse"],
        "selection_score": initial_validation["selection_score"],
    }]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        torch,
        args.output,
        elastic,
        fixed,
        widths=widths,
        validation_widths=validation_widths,
        retention_weight=args.selection_retention_weight,
        epoch=0,
        score=best,
        validation=initial_validation,
    )
    print(json.dumps({"epoch": 0, **initial_validation, "best": best, "stale": 0}), flush=True)
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        elastic.train()
        total_loss = 0.0
        batches = 0
        for batch_x, batch_u in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_u = batch_u.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            maximum_config = elastic.max_config
            full_prediction = elastic(batch_x, maximum_config)
            with torch.no_grad():
                immutable_teacher = fixed.normalized_forward(batch_x)
            configs = [maximum_config, SubnetConfig.uniform(widths[0], elastic.hidden_layers)]
            configs.extend(
                SubnetConfig.uniform(random.choice(widths[1:-1]), elastic.hidden_layers)
                for _ in range(args.random_subnets)
            )
            full_task_loss = loss_function(full_prediction, batch_u)
            full_retention_loss = loss_function(full_prediction, immutable_teacher)
            weighted_loss = (
                args.full_loss_weight * full_task_loss
                + args.full_kd_weight * full_retention_loss
            )
            loss_weight = args.full_loss_weight + args.full_kd_weight
            for config in configs[1:]:
                prediction = elastic(batch_x, config)
                task_loss = loss_function(prediction, batch_u)
                kd_loss = loss_function(prediction, immutable_teacher)
                weighted_loss = weighted_loss + task_loss + args.kd_weight * kd_loss
                loss_weight += 1.0 + args.kd_weight
            loss = weighted_loss / loss_weight
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        validation = _validation_score(
            torch,
            elastic,
            fixed,
            val_x_t,
            val_u_t,
            validation_widths,
            args.batch_size,
            args.selection_retention_weight,
        )
        score = validation["selection_score"]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "validation_family_mse": validation["family_mse"],
            "validation_full_vs_fixed_mse": validation["full_vs_fixed_mse"],
            "selection_score": score,
        }
        history.append(row)
        improved = score < best
        if improved:
            best = score
            stale = 0
            _save_checkpoint(
                torch,
                args.output,
                elastic,
                fixed,
                widths=widths,
                validation_widths=validation_widths,
                retention_weight=args.selection_retention_weight,
                epoch=epoch,
                score=best,
                validation=validation,
            )
        else:
            stale += 1
        print(json.dumps({**row, "best": best, "stale": stale}), flush=True)
        if stale >= args.patience:
            break

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(args.output.resolve()),
        "best_selection_score": best,
        "epochs_completed": len(history) - 1,
        "elapsed_seconds": time.perf_counter() - start_time,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
