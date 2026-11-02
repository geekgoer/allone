"""Reproduce the immutable fixed-controller offline metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elastic_supernet_test_a.evaluation.common import load_split, regression_metrics
from elastic_supernet_test_a.models.baseline_mlp import load_fixed_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("elastic_supernet_test_a/results/baseline_fixed_model.json"))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    controller = load_fixed_controller(torch, args.model, args.normalization, device=device)
    x, target = load_split(args.dataset, max_samples=args.max_samples)
    metrics = regression_metrics(torch, controller, x, target, batch_size=args.batch_size, device=device)
    result: dict[str, Any] = {
        "model": controller.info.__dict__,
        "dataset": str(args.dataset.resolve()),
        "device": device,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

