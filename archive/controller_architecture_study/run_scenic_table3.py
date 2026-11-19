"""Train missing SCENIC Table III plain MLPs sequentially and restartably."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_controller import train


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/scenic_table3_v1.json"
DEFAULT_RESULTS = HERE / "results/scenic_table3_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.config.read_text(encoding="utf-8"))
    architectures = source["architectures"]
    shared = {key: value for key, value in source.items()
              if key not in {"paper", "paper_source", "architecture_note", "architectures",
                             "already_trained_in_previous_coarse_sweep"}}
    previous = {tuple(item) for item in source["already_trained_in_previous_coarse_sweep"]}

    for item in architectures:
        layers = int(item["hidden_layers"])
        width = int(item["hidden_width"])
        name = f"plain_mlp_h{layers}_w{width}_seed{source['seed']}"
        target = args.results_dir / name
        old_target = HERE / "results" / name
        if (layers, width) in previous and (old_target / "offline_metrics.json").exists():
            print(f"Already trained in coarse sweep: {name} -> {old_target}", flush=True)
            continue
        state = "complete" if (target / "offline_metrics.json").exists() else "resume" if (target / "config.json").exists() else "new"
        print(f"SCENIC Table III index={item['index']} L={item['paper_L']} {name}: {state} -> {target}", flush=True)
        if args.dry_run or state == "complete":
            continue
        config = {**shared, "hidden_layers": layers, "hidden_width": width}
        train(config, target, resume=state == "resume", device=args.device)

    print(f"SCENIC Table III sweep complete: {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
