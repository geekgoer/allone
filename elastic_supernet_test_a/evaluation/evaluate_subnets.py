"""Evaluate prefix-sliced elastic subnets and write MAC/error Pareto artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from elastic_supernet_test_a.analysis.compute_cost import mlp_active_parameters, mlp_macs
from elastic_supernet_test_a.analysis.pareto import lower_is_better_frontier
from elastic_supernet_test_a.evaluation.common import load_split, regression_metrics
from elastic_supernet_test_a.models.baseline_mlp import load_fixed_controller
from elastic_supernet_test_a.models.elastic_mlp import ElasticController, ElasticMLP, SubnetConfig


DEFAULT_WIDTHS = "16,32,48,64,96,128,160,192,256,320,384,448,512"


def _torch() -> Any:
    import torch
    return torch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]], frontier: list[dict[str, Any]], title: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter([row["macs"] for row in rows], [row["mse"] for row in rows], label="elastic subnets")
    axis.plot([row["macs"] for row in frontier], [row["mse"] for row in frontier], "o-", label="Pareto frontier")
    axis.set_xlabel("Active MAC count")
    axis.set_ylabel("Physical control MSE (lower is better)")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = load_fixed_controller(torch, args.model, args.normalization, device=device)
    elastic_mlp = ElasticMLP.from_fixed_controller(fixed).to(device).eval()
    if args.elastic_checkpoint:
        payload = torch.load(args.elastic_checkpoint, map_location=device, weights_only=False)
        elastic_mlp.load_state_dict(payload["elastic_state_dict"])
    elastic = ElasticController(elastic_mlp, fixed).to(device).eval()
    x, target = load_split(args.dataset, max_samples=args.max_samples)

    probe = torch.from_numpy(x[: min(1024, len(x))]).to(device)
    with torch.inference_mode():
        fixed_output = fixed(probe)
        maximum_output = elastic(probe, elastic_mlp.max_config)
    max_difference = float((fixed_output - maximum_output).abs().max().item())
    if args.elastic_checkpoint is None and max_difference > args.match_tolerance:
        raise RuntimeError(f"maximum elastic subnet mismatch: {max_difference}")

    widths = sorted(set(int(value) for value in args.widths.split(",")))
    if not widths or widths[-1] > fixed.info.width:
        raise ValueError("candidate widths must not exceed the baseline width")
    if fixed.info.width not in widths:
        widths.append(fixed.info.width)
        widths.sort()

    rows: list[dict[str, Any]] = []
    for width in widths:
        config = SubnetConfig.uniform(width, fixed.info.hidden_layers)
        metrics = regression_metrics(
            torch,
            lambda batch, selected=config: elastic(batch, selected),
            x,
            target,
            batch_size=args.batch_size,
            device=device,
            reference_predict=fixed,
        )
        row = {
            "config": config.label,
            "uniform_width": width,
            "hidden_layers": fixed.info.hidden_layers,
            "macs": mlp_macs(fixed.info.input_dim, config.widths, fixed.info.output_dim),
            "active_parameters": mlp_active_parameters(fixed.info.input_dim, config.widths, fixed.info.output_dim),
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "max_abs_error": metrics["max_abs_error"],
            "mse_vs_fixed_full": metrics["mse_vs_fixed_full"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    frontier = lower_is_better_frontier(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = "after_finetune" if args.elastic_checkpoint else "before_finetune"
    subnets_csv = args.output_dir / f"subnets_{phase}.csv"
    pareto_csv = args.output_dir / f"pareto_{phase}.csv"
    figure = args.figure_dir / f"quality_vs_mac_{phase}.png"
    _write_csv(subnets_csv, rows)
    _write_csv(pareto_csv, frontier)
    plot_error = _plot(figure, rows, frontier, f"Elastic MLP {phase.replace('_', ' ')}")
    summary = {
        "phase": phase,
        "baseline": fixed.info.__dict__,
        "dataset": str(args.dataset.resolve()),
        "device": device,
        "candidate_count": len(rows),
        "pareto_count": len(frontier),
        "max_subnet_vs_fixed_max_abs": max_difference,
        "subnets_csv": str(subnets_csv.resolve()),
        "pareto_csv": str(pareto_csv.resolve()),
        "figure": str(figure.resolve()) if plot_error is None else None,
        "plot_error": plot_error,
    }
    (args.output_dir / f"summary_{phase}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("elastic_supernet_test_a/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("elastic_supernet_test_a/figures"))
    parser.add_argument("--elastic-checkpoint", type=Path)
    parser.add_argument("--widths", default=DEFAULT_WIDTHS)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--match-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2), flush=True)


if __name__ == "__main__":
    main()

