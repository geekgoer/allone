"""Evaluate continuous MPC control imitation in normalized and physical units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models.factory import build_model


OUTPUT_NAMES = (
    "collective_thrust_acceleration", "body_rate_x", "body_rate_y", "body_rate_z"
)


def load_normalization(data_dir: Path) -> dict[str, np.ndarray]:
    payload = json.loads((data_dir / "normalization.json").read_text(encoding="utf-8"))
    values = {key: np.asarray(payload[key], dtype=np.float32) for key in ("x_mean", "x_std", "u_mean", "u_std")}
    if values["x_mean"].shape not in ((12,), (237,), (300,)) or values["x_std"].shape != values["x_mean"].shape:
        raise ValueError("dataset must have 12, 237 or 300 input features")
    if values["u_mean"].shape != (4,) or values["u_std"].shape != (4,):
        raise ValueError("dataset must have 4 output features")
    if any(np.any(values[key] <= 0) for key in ("x_std", "u_std")):
        raise ValueError("normalization standard deviations must be positive")
    return values


def load_split(data_dir: Path, split: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
        x = np.asarray(archive["X"][:limit], dtype=np.float32)
        u = np.asarray(archive["U"][:limit], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] not in (12, 237, 300) or u.shape != (len(x), 4) or len(x) == 0:
        raise ValueError(f"unexpected {split} shapes: {x.shape}, {u.shape}")
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError(f"non-finite values in {split}")
    return x, u


def error_metrics(predicted_normalized: np.ndarray, target: np.ndarray, norm: dict) -> dict:
    normalized_target = (target - norm["u_mean"]) / norm["u_std"]
    normalized_error = np.asarray(predicted_normalized, dtype=np.float32) - normalized_target
    physical_error = normalized_error * norm["u_std"]
    absolute = np.abs(physical_error)
    return {
        "samples": int(len(target)),
        "normalized_mse": float(np.mean(normalized_error**2)),
        "physical_mse": float(np.mean(physical_error**2)),
        "physical_rmse": float(np.sqrt(np.mean(physical_error**2))),
        "p95_error": float(np.percentile(absolute, 95)),
        "p99_error": float(np.percentile(absolute, 99)),
        "per_output_mse": {name: float(value) for name, value in zip(OUTPUT_NAMES, np.mean(physical_error**2, axis=0))},
        "per_output_rmse": {name: float(value) for name, value in zip(OUTPUT_NAMES, np.sqrt(np.mean(physical_error**2, axis=0)))},
        "p95_abs_error": {name: float(value) for name, value in zip(OUTPUT_NAMES, np.percentile(absolute, 95, axis=0))},
        "p99_abs_error": {name: float(value) for name, value in zip(OUTPUT_NAMES, np.percentile(absolute, 99, axis=0))},
        "max_abs_error": {name: float(value) for name, value in zip(OUTPUT_NAMES, np.max(absolute, axis=0))},
    }


def evaluate(model, data_dir: Path, split: str, *, device: str, batch_size: int = 4096, limit: int | None = None) -> dict:
    import torch

    norm = load_normalization(data_dir)
    x, u = load_split(data_dir, split, limit)
    if x.shape[1] != len(norm["x_mean"]):
        raise ValueError("split and normalization input dimensions disagree")
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            chunk = (x[start:start + batch_size] - norm["x_mean"]) / norm["x_std"]
            predictions.append(model(torch.from_numpy(chunk).to(device)).cpu().numpy())
    predicted = np.concatenate(predictions)
    metrics = error_metrics(predicted, u, norm)
    with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
        if "family_id" in archive:
            families = archive["family_id"][:limit]
            metrics["by_family_id"] = {str(int(family)): error_metrics(predicted[families == family], u[families == family], norm)
                                       for family in np.unique(families)}
    return metrics


def load_checkpoint(path: Path, *, device: str):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise ValueError("checkpoint is missing model_state")
    if "hidden_layers" in payload:
        hidden_layers = int(payload["hidden_layers"])
        hidden_width = int(payload["hidden_width"])
    else:
        hidden_layers = int(payload["depth"]) - 1
        hidden_width = int(payload["width"])
    model = build_model(torch, model_type=payload.get("model_type", "plain_mlp"),
                        input_dim=int(payload.get("input_dim", 12)), output_dim=4, hidden_layers=hidden_layers, hidden_width=hidden_width)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(load_checkpoint(args.checkpoint, device=args.device), args.data_dir, args.split, device=args.device, batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(metrics, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
