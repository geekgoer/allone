"""Export ONE shared FP32 model, packed causal blocks and 136 configuration metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..eval_offline import error_metrics
from .common import (ROOT, DEFAULT_RUN, atomic_json, evaluate_grid, grid, load_supernet,
                     sha256, verify_assets)
from .model import IncrementalContext, NETWORK_CONTRACT, PhysicalController


def pack_weights(model):
    g = model.group_width
    weights = {"stem_weight": model.stem.weight.detach().cpu().numpy(),
               "stem_bias": model.stem.bias.detach().cpu().numpy()}
    for d, block in enumerate(model.blocks, 2):
        weights[f"layer{d}_bias"] = block.bias.detach().cpu().numpy()
        for out in range(len(model.widths)):
            for src in range(out + 1):
                weights[f"layer{d}_g{out}_from{src}"] = block.weight[out*g:(out+1)*g, src*g:(src+1)*g].detach().cpu().numpy()
    for d, head in zip(model.depths, model.heads):
        weights[f"head{d}_weight"] = head.weight.detach().cpu().numpy()
        weights[f"head{d}_bias"] = head.bias.detach().cpu().numpy()
    return weights


def numpy_forward(raw, depth, width, weights, spec, norm):
    """Framework-independent packed FP32 reference, including raw/physical conversion."""
    x = (raw - norm["x_mean"]) / norm["x_std"]
    g, groups = spec["group_width"], width // spec["group_width"]
    h = [np.maximum(x @ weights["stem_weight"][i*g:(i+1)*g].T + weights["stem_bias"][i*g:(i+1)*g], 0)
         for i in range(groups)]
    for d in range(2, depth + 1):
        next_h = []
        for out in range(groups):
            branch = np.broadcast_to(weights[f"layer{d}_bias"][out*g:(out+1)*g], h[out].shape).copy()
            for src in range(out + 1):
                branch += h[src] @ weights[f"layer{d}_g{out}_from{src}"].T
            next_h.append(np.maximum(h[out] + np.float32(spec["residual_scale"]) * branch, 0))
        h = next_h
    output = np.broadcast_to(weights[f"head{depth}_bias"], (len(raw), 4)).copy()
    for i in range(groups):
        output += h[i] @ weights[f"head{depth}_weight"][:, i*g:(i+1)*g].T
    return output * norm["u_std"] + norm["u_mean"]


def export(checkpoint, output, device="cpu", golden_count=32):
    if golden_count < 1:
        raise ValueError("golden_count must be positive")
    model, saved = load_supernet(checkpoint, device)
    config = saved["config"]
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    norm, data, families = verify_assets(config)
    output = output.resolve()
    protected = [(ROOT / config["data_dir"]).resolve(), (ROOT / config["teacher_checkpoint"]).resolve().parent]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("export cannot overwrite frozen data/teacher")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(checkpoint)
    marker = output / "export_report.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("export directory belongs to another checkpoint; choose a new directory")
    metadata = json.loads((ROOT / config["data_dir"] / "normalization.json").read_text(encoding="utf-8"))
    atomic_json(output / "normalization.json", metadata)
    configs = grid(model)
    spec = {"network_contract": NETWORK_CONTRACT, "input_contract": config["input_contract"],
            "input_dim": config["input_dim"], "output_dim": 4, "group_width": model.group_width,
            "depth_choices": model.depths, "width_choices": model.widths, "configuration_count": len(configs),
            "hidden_depth_includes_stem": True, "group_indices_are_zero_based": True,
            "causal_dependency": "output_group g reads input_groups 0..g; no upper blocks",
            "readout": "sum of group contributions, bias added once", "residual_scale": model.residual_scale,
            "normalization": "scripted controller accepts RAW input and returns PHYSICAL output",
            "incremental_context": "same sample + fixed model; cache ALL completed layers/groups",
            "allocated_parameters": sum(p.numel() for p in model.parameters()),
            "effective_parameters": (sum(p.numel() for p in model.stem.parameters())
                + len(model.blocks)*(int(model.causal_mask.sum()) + model.max_width)
                + sum(p.numel() for p in model.heads.parameters())),
            "dense_mask_training_is_not_sparse_GPU_execution": True,
            "checkpoint_sha256": checkpoint_hash, "teacher_sha256": config["teacher_sha256"],
            "test_read": False}
    weights = pack_weights(model)
    path = output / "weights_fp32.npz"
    with path.with_name(path.name + ".tmp").open("wb") as handle:
        np.savez(handle, **weights)
    path.with_name(path.name + ".tmp").replace(path)
    spec["packed_parameter_count"] = sum(v.size for v in weights.values())
    spec["group_layout_sha256"] = hashlib.sha256(json.dumps(
        {k: list(v.shape) for k, v in weights.items()}, sort_keys=True).encode()).hexdigest()
    atomic_json(output / "model_spec.json", spec)
    print("Exporting shared TorchScript and checking golden vectors...", flush=True)
    model.cpu()
    scripted = torch.jit.script(PhysicalController(model, norm).eval())
    temporary = output / "controller_scripted.pt.tmp"
    torch.jit.save(scripted, temporary)
    temporary.replace(output / "controller_scripted.pt")
    count = min(golden_count, len(data["validation"][0]))
    indices = np.linspace(0, len(data["validation"][0])-1, count, dtype=int)
    x_norm = data["validation"][0][indices]
    raw = x_norm.numpy() * norm["x_std"] + norm["x_mean"]
    gold, reference = [], []
    scripted_error, cache_error = 0., 0.
    with torch.inference_mode():
        for d, w in configs:
            physical = model(x_norm, d, w).numpy() * norm["u_std"] + norm["u_mean"]
            actual = scripted(torch.from_numpy(raw), d, w).numpy()
            gold.append(physical)
            reference.append(numpy_forward(raw, d, w, weights, spec, norm))
            scripted_error = max(scripted_error, float(np.max(np.abs(actual - physical))))
        for order in ([(model.max_depth, model.widths[0]), (model.max_depth, model.max_width)],
                      [(model.depths[0], model.max_width), (model.max_depth, model.max_width)]):
            cache = IncrementalContext(model, x_norm[:2], "golden", checkpoint_hash)
            for d, w in order:
                candidate, _ = cache.grow(d, w, sample_id="golden", model_version=checkpoint_hash)
            cache_error = max(cache_error, float((candidate - model(x_norm[:2])).abs().max()))
    gold, reference = np.asarray(gold), np.asarray(reference)
    numpy_error = float(np.max(np.abs(gold - reference)))
    if (not np.allclose(gold, reference, atol=5e-5, rtol=2e-5)
            or scripted_error > 5e-5 or cache_error > 5e-5):
        raise RuntimeError(f"FP32 export mismatch: numpy={numpy_error}, script={scripted_error}, cache={cache_error}")
    with (output / "golden_vectors.npz.tmp").open("wb") as handle:
        np.savez(handle, X_raw=raw, X_normalized=x_norm.numpy(), configurations=np.asarray(configs),
                 outputs_physical=gold, outputs_numpy_physical=reference, validation_row_indices=indices)
    (output / "golden_vectors.npz.tmp").replace(output / "golden_vectors.npz")
    model.to(device)
    rows = []
    for split in ("train", "validation"):
        print(f"Offline {split}: {len(data[split][0])} rows, all {len(configs)} configurations; test NOT read", flush=True)
        _, predictions = evaluate_grid(model, *data[split], config["evaluation_batch_size"], return_predictions=True)
        targets = data[split][1].numpy() * norm["u_std"] + norm["u_mean"]
        for d, w in configs:
            if split == "train":
                rows.append({"depth": d, "width": w, **model.complexity(d, w)})
            row = next(row for row in rows if row["depth"] == d and row["width"] == w)
            row[split] = error_metrics(predictions[(d, w)], targets, norm)
            row[split]["by_family_id"] = {str(int(f)): error_metrics(predictions[(d,w)][families[split] == f],
                targets[families[split] == f], norm) for f in np.unique(families[split])}
    atomic_json(output / "offline_metrics.json", {"test_read": False, "checkpoint_epoch": saved["epoch"], "configurations": rows})
    report = {"status": "PASS", "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": saved["epoch"],
              "configurations": len(configs), "golden_count": count, "numpy_max_abs_error": numpy_error,
              "scripted_max_abs_error": scripted_error, "incremental_max_abs_error": cache_error,
              "maximum_quality_goal_met": saved["best_key"][0] == 0, "test_read": False,
              "artifact_sha256": {name: sha256(output/name) for name in ("weights_fp32.npz", "model_spec.json",
                  "normalization.json", "controller_scripted.pt", "golden_vectors.npz", "offline_metrics.json")}}
    atomic_json(marker, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN/"best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN/"export")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--golden-count", type=int, default=32)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.device, args.golden_count)


if __name__ == "__main__":
    main()
