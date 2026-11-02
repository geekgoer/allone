"""Finalize a completed SCENIC run without retraining it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compute_complexity import checkpoint_size, complexity
from .eval_offline import evaluate, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = load_checkpoint(checkpoint, device="cpu")
    metrics = {
        split: evaluate(model, args.data_dir, split, device="cpu")
        for split in ("validation", "test")
    }
    (run_dir / "offline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    cost = complexity(
        12,
        4,
        int(config["hidden_layers"]),
        int(config["hidden_width"]),
    )
    cost["elementwise_residual_adds"] = 0
    cost["elementwise_residual_multiplies"] = 0
    cost["checkpoint_size_bytes"] = checkpoint_size(checkpoint)
    (run_dir / "complexity.json").write_text(
        json.dumps(cost, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["test"], indent=2), flush=True)
    print(f"Finalized: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
