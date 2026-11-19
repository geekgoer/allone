"""Evaluate a trained DNN on validation or test samples without retraining.

Both physical-unit and normalized-output errors are reported.  Inference is
batched so the same evaluator can safely audit the 34x1024 checkpoint and its
TorchScript export on laptop GPUs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dnn_model_utils import build_model, load_normalization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dnn_dataset_full_route_v2"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional deterministic prefix for quick consistency checks",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    import torch

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    try:
        model = torch.jit.load(str(args.model), map_location=args.device)
        model_format = "torchscript"
        checkpoint = None
    except RuntimeError:
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(
                "model must be TorchScript or a checkpoint containing model_state"
            )
        input_dim = int(checkpoint.get("input_dim", 12))
        model = build_model(
            torch,
            depth=int(checkpoint["depth"]),
            width=int(checkpoint["width"]),
            input_dim=input_dim,
        )
        model.load_state_dict(checkpoint["model_state"])
        model_format = "checkpoint"
    normalization = load_normalization(args.data_dir / "normalization.json")
    input_dim = int(normalization["x_mean"].size)
    model.to(args.device).eval()
    with np.load(args.data_dir / f"{args.split}.npz", allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        target = np.asarray(archive["U"], dtype=np.float32)
    if args.max_samples is not None:
        x = x[: args.max_samples]
        target = target[: args.max_samples]
    if x.shape[1:] != (input_dim,):
        raise ValueError(f"test input shape {x.shape} does not match model input {input_dim}")
    if target.shape != (len(x), 4):
        raise ValueError(f"test target shape must be (N, 4), got {target.shape}")
    output_dimensions = int(target.shape[1])
    physical_squared = np.zeros(output_dimensions, dtype=np.float64)
    physical_absolute = np.zeros(output_dimensions, dtype=np.float64)
    normalized_squared = np.zeros(output_dimensions, dtype=np.float64)
    normalized_absolute = np.zeros(output_dimensions, dtype=np.float64)
    physical_max = 0.0
    normalized_max = 0.0
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            stop = min(start + args.batch_size, len(x))
            x_norm = (
                (x[start:stop] - normalization["x_mean"])
                / normalization["x_std"]
            ).astype(np.float32, copy=False)
            predicted_norm = (
                model(torch.from_numpy(x_norm).to(args.device)).detach().cpu().numpy()
            )
            prediction = (
                predicted_norm * normalization["u_std"] + normalization["u_mean"]
            )
            physical_error = prediction - target[start:stop]
            target_norm = (
                (target[start:stop] - normalization["u_mean"])
                / normalization["u_std"]
            )
            normalized_error = predicted_norm - target_norm
            physical_squared += np.sum(
                physical_error.astype(np.float64) ** 2, axis=0
            )
            physical_absolute += np.sum(
                np.abs(physical_error.astype(np.float64)), axis=0
            )
            normalized_squared += np.sum(
                normalized_error.astype(np.float64) ** 2, axis=0
            )
            normalized_absolute += np.sum(
                np.abs(normalized_error.astype(np.float64)), axis=0
            )
            physical_max = max(physical_max, float(np.max(np.abs(physical_error))))
            normalized_max = max(normalized_max, float(np.max(np.abs(normalized_error))))
    physical_mse_per_output = physical_squared / len(x)
    physical_mae_per_output = physical_absolute / len(x)
    normalized_mse_per_output = normalized_squared / len(x)
    normalized_mae_per_output = normalized_absolute / len(x)
    physical_mse = float(np.mean(physical_mse_per_output))
    normalized_mse = float(np.mean(normalized_mse_per_output))
    result = {
        "split": args.split,
        "model": str(args.model.resolve()),
        "model_format": model_format,
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "input_dimensions": input_dim,
        "samples": int(len(x)),
        "mse": physical_mse,
        "mae": float(np.mean(physical_mae_per_output)),
        "rmse": float(np.sqrt(physical_mse)),
        "max_abs_error": physical_max,
        "mse_per_output": physical_mse_per_output.tolist(),
        "mae_per_output": physical_mae_per_output.tolist(),
        "normalized": {
            "mse": normalized_mse,
            "mae": float(np.mean(normalized_mae_per_output)),
            "rmse": float(np.sqrt(normalized_mse)),
            "max_abs_error": normalized_max,
            "mse_per_output": normalized_mse_per_output.tolist(),
            "mae_per_output": normalized_mae_per_output.tolist(),
        },
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
