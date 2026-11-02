"""Supplementary GPU benchmark for fixed-pool and elastic-subnet switching.

MAC/error remains the primary Test A evidence. This benchmark only measures
the software/GPU behavior of already-resident controllers; it is not an FPGA
latency predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable

import numpy as np

from elastic_supernet_test_a.models.baseline_mlp import load_fixed_controller
from elastic_supernet_test_a.models.elastic_mlp import ElasticController, ElasticMLP, SubnetConfig


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p95_us": _percentile(values, 95),
        "p99_us": _percentile(values, 99),
        "min_us": min(values),
        "max_us": max(values),
    }


def _parameter_bytes(module: Any) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def _timed_call(torch: Any, function: Callable[[], Any], *, cuda: bool) -> tuple[float, float]:
    wall_start = time.perf_counter_ns()
    if cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        torch.cuda.synchronize()
        device_us = float(start.elapsed_time(end) * 1000.0)
    else:
        output = function()
        device_us = float("nan")
    # Force the returned tensor to remain live through synchronization.
    if output is None:
        raise RuntimeError("controller returned no output")
    wall_us = (time.perf_counter_ns() - wall_start) / 1000.0
    return device_us, wall_us


def _run_sequence(
    torch: Any,
    options: dict[str, Callable[[], Any]],
    sequence: list[str],
    *,
    warmup: int,
    cuda: bool,
) -> dict[str, Any]:
    names = list(options)
    for index in range(warmup):
        options[names[index % len(names)]]()
    if cuda:
        torch.cuda.synchronize()

    device_by_name = {name: [] for name in names}
    wall_by_name = {name: [] for name in names}
    all_device: list[float] = []
    all_wall: list[float] = []
    transitions = 0
    previous: str | None = None
    for name in sequence:
        if previous is not None and name != previous:
            transitions += 1
        device_us, wall_us = _timed_call(torch, options[name], cuda=cuda)
        if cuda:
            device_by_name[name].append(device_us)
            all_device.append(device_us)
        wall_by_name[name].append(wall_us)
        all_wall.append(wall_us)
        previous = name
    return {
        "sequence_length": len(sequence),
        "transitions": transitions,
        "transition_fraction": transitions / max(1, len(sequence) - 1),
        "device_all": _summary(all_device) if cuda else None,
        "wall_all": _summary(all_wall),
        "device_by_option": {
            name: _summary(values) for name, values in device_by_name.items() if values
        },
        "wall_by_option": {
            name: _summary(values) for name, values in wall_by_name.items() if values
        },
    }


def _static_medians(
    torch: Any,
    options: dict[str, Callable[[], Any]],
    *,
    iterations: int,
    warmup: int,
    cuda: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, function in options.items():
        result[name] = _run_sequence(
            torch,
            {name: function},
            [name] * iterations,
            warmup=warmup,
            cuda=cuda,
        )
    return result


def _switch_penalty(switched: dict[str, Any], static: dict[str, Any], metric: str) -> dict[str, float]:
    field = "device_by_option" if metric == "device" else "wall_by_option"
    result: dict[str, float] = {}
    for name, summary in switched[field].items():
        static_summary = static[name][field][name]
        result[name] = float(summary["median_us"] - static_summary["median_us"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-pool-manifest", type=Path, required=True)
    parser.add_argument("--elastic-model", type=Path, required=True)
    parser.add_argument("--elastic-normalization", type=Path, required=True)
    parser.add_argument("--elastic-checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--elastic-widths", default="64,128,256,384,512")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("elastic_supernet_test_a/results/gpu_switching.json"))
    args = parser.parse_args()

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested GPU switching benchmark")
    cuda = args.device.startswith("cuda")
    with np.load(args.dataset, allow_pickle=False) as archive:
        sample = torch.from_numpy(np.asarray(archive["X"][0:1], dtype=np.float32)).to(args.device)

    manifest = json.loads(args.fixed_pool_manifest.read_text(encoding="utf-8"))
    fixed_options: dict[str, Callable[[], Any]] = {}
    fixed_models: list[Any] = []
    for item in manifest["models"]:
        controller = load_fixed_controller(
            torch, Path(item["checkpoint"]), Path(item["normalization"]), device=args.device
        )
        fixed_models.append(controller)
        fixed_options[str(item["name"])] = lambda selected=controller: selected(sample)

    fixed_for_elastic = load_fixed_controller(
        torch, args.elastic_model, args.elastic_normalization, device=args.device
    )
    elastic_mlp = ElasticMLP.from_fixed_controller(fixed_for_elastic).to(args.device).eval()
    if args.elastic_checkpoint:
        payload = torch.load(args.elastic_checkpoint, map_location=args.device, weights_only=False)
        elastic_mlp.load_state_dict(payload["elastic_state_dict"])
    elastic = ElasticController(elastic_mlp, fixed_for_elastic).to(args.device).eval()
    elastic_options: dict[str, Callable[[], Any]] = {}
    for width in sorted(set(int(value) for value in args.elastic_widths.split(","))):
        config = SubnetConfig.uniform(width, elastic_mlp.hidden_layers)
        elastic_options[config.label] = lambda selected=config: elastic(sample, selected)

    rng = random.Random(args.seed)
    fixed_sequence = [rng.choice(list(fixed_options)) for _ in range(args.iterations)]
    elastic_sequence = [rng.choice(list(elastic_options)) for _ in range(args.iterations)]
    fixed_static = _static_medians(
        torch, fixed_options, iterations=args.iterations, warmup=args.warmup, cuda=cuda
    )
    elastic_static = _static_medians(
        torch, elastic_options, iterations=args.iterations, warmup=args.warmup, cuda=cuda
    )
    fixed_switched = _run_sequence(
        torch, fixed_options, fixed_sequence, warmup=args.warmup, cuda=cuda
    )
    elastic_switched = _run_sequence(
        torch, elastic_options, elastic_sequence, warmup=args.warmup, cuda=cuda
    )
    result = {
        "scope": "supplementary resident-model GPU switching benchmark; not an FPGA predictor",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if cuda else None,
        "iterations_per_static_option": args.iterations,
        "fixed_pool": {
            "resident_models": list(fixed_options),
            "resident_parameter_bytes": sum(_parameter_bytes(item.model) for item in fixed_models),
            "static": fixed_static,
            "switched": fixed_switched,
            "switch_penalty_device_median_us": _switch_penalty(fixed_switched, fixed_static, "device") if cuda else None,
            "switch_penalty_wall_median_us": _switch_penalty(fixed_switched, fixed_static, "wall"),
        },
        "elastic_supernet": {
            "resident_parameter_sets": 1,
            "resident_parameter_bytes": _parameter_bytes(elastic_mlp.module),
            "options": list(elastic_options),
            "static": elastic_static,
            "switched": elastic_switched,
            "switch_penalty_device_median_us": _switch_penalty(elastic_switched, elastic_static, "device") if cuda else None,
            "switch_penalty_wall_median_us": _switch_penalty(elastic_switched, elastic_static, "wall"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "fixed_switched_device": fixed_switched["device_all"],
        "elastic_switched_device": elastic_switched["device_all"],
        "fixed_switch_penalty_device_median_us": result["fixed_pool"]["switch_penalty_device_median_us"],
        "elastic_switch_penalty_device_median_us": result["elastic_supernet"]["switch_penalty_device_median_us"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
