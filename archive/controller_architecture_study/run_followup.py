"""Run controlled seed repeats and a matched-size residual ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_controller import train


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/followup_v1.json"
DEFAULT_RESULTS = HERE / "results/followup_v1"


def build_jobs(followup: dict, protocol: dict, phase: str) -> list[tuple[str, dict]]:
    shared = {key: value for key, value in protocol.items() if key not in ("architectures", "seed", "model_type", "residual")}
    jobs = []
    groups = []
    if phase in ("repeats", "all"):
        groups.append(("plain_mlp", followup["plain_seed_repeats"]))
    if phase in ("residual", "all"):
        groups.append(("residual_mlp", followup["residual_tests"]))
    for model_type, cases in groups:
        for case in cases:
            for seed in case["seeds"]:
                config = {**shared, "model_type": model_type, "residual": model_type == "residual_mlp",
                          "hidden_layers": int(case["hidden_layers"]), "hidden_width": int(case["hidden_width"]), "seed": int(seed)}
                name = f"{model_type}_h{config['hidden_layers']}_w{config['hidden_width']}_seed{seed}"
                jobs.append((name, config))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--phase", choices=("repeats", "residual", "all"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="two short jobs in a separate smoke directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    followup = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads((args.config.parent / followup["protocol"]).read_text(encoding="utf-8"))
    jobs = build_jobs(followup, protocol, args.phase)
    results = args.results_dir / "smoke" if args.smoke else args.results_dir
    if args.smoke:
        jobs = [job for job in jobs if job[0] in ("plain_mlp_h4_w128_seed1", "residual_mlp_h10_w512_seed0")]
        jobs = [(name, {**config, "max_epochs": 2, "min_epochs": 1, "patience": 2,
                        "train_limit": 2048, "validation_limit": 1024}) for name, config in jobs]
    for name, config in jobs:
        target = results / name
        state = "complete" if (target / "offline_metrics.json").exists() else "resume" if (target / "config.json").exists() else "new"
        print(f"{name}: {state} -> {target}", flush=True)
        if not args.dry_run and state != "complete":
            train(config, target, resume=state == "resume", device=args.device)
    print(f"Phase {args.phase}: {len(jobs)} jobs listed in {results}", flush=True)


if __name__ == "__main__":
    main()
