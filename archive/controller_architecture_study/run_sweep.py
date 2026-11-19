"""Run the fixed M1 coarse grid sequentially; each architecture is restartable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_controller import train


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/coarse_sweep.json"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="2 epochs and small subsets; separate smoke directory")
    args = parser.parse_args()
    shared = json.loads(args.config.read_text(encoding="utf-8"))
    architectures = shared.pop("architectures")
    results = args.results_dir / "smoke" if args.smoke else args.results_dir
    if args.smoke:
        architectures = [(2, 64), (4, 128)]
        shared.update(max_epochs=2, min_epochs=1, patience=2, train_limit=2048, validation_limit=1024)
    for layers, width in architectures:
        config = {**shared, "hidden_layers": layers, "hidden_width": width}
        target = results / f"plain_mlp_h{layers}_w{width}_seed{config['seed']}"
        if (target / "offline_metrics.json").exists():
            print(f"Already complete: {target}", flush=True)
            continue
        print(f"Starting: {target}", flush=True)
        train(config, target, resume=(target / "config.json").exists(), device=args.device)
    print(f"Sweep complete: {results}", flush=True)


if __name__ == "__main__":
    main()
