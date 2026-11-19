"""Three-stage FP32 training. Re-running --resume continues the last saved epoch."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .common import (ROOT, DEFAULT_CONFIG, DEFAULT_RUN, atomic_json, atomic_torch, evaluate_grid,
                     grid, load_teacher, model_from_config, restore_rng, rng_state, sha256, verify_assets)
from .model import NETWORK_CONTRACT


def stage_at(config, epoch):
    offset = 0
    for i, stage in enumerate(config["stages"]):
        if epoch <= offset + stage["epochs"]:
            return i, epoch - offset
        offset += stage["epochs"]
    raise ValueError("epoch exceeds the fixed training budget")


def optimizer_for(model, stage, config):
    name = stage["name"]
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    backbone = list(model.stem.parameters()) + list(model.blocks.parameters()) + list(model.heads[-1].parameters())
    new_heads = [p for head in model.heads[:-1] for p in head.parameters()]
    groups = []
    if name in ("adapt", "joint"):
        for p in backbone:
            p.requires_grad_(True)
        groups.append({"params": backbone, "lr": stage["backbone_lr"], "base_lr": stage["backbone_lr"]})
    if name in ("heads", "joint"):
        for p in new_heads:
            p.requires_grad_(True)
        groups.append({"params": new_heads, "lr": stage["head_lr"], "base_lr": stage["head_lr"]})
    if name == "heads" and stage.get("maximum_head_lr", 0) > 0:
        for p in model.heads[-1].parameters():
            p.requires_grad_(True)
        groups.append({"params": list(model.heads[-1].parameters()),
                       "lr": stage["maximum_head_lr"], "base_lr": stage["maximum_head_lr"]})
    return torch.optim.Adam(groups, weight_decay=config["weight_decay"], foreach=False)


def sample_configs(model, stage, sampler, include_maximum_head=False):
    largest = (model.max_depth, model.max_width)
    if stage == "adapt":
        return [largest], []
    depths = model.depths[:-1] if stage == "heads" and not include_maximum_head else model.depths
    configs = {(depths[0], model.widths[0])}
    if stage == "joint" or include_maximum_head:
        configs.add(largest)
    d = sampler.randrange(len(depths) - 1)
    width = sampler.choice(model.widths)
    depth_pair = ((depths[d], width), (depths[d+1], width))
    w = sampler.randrange(len(model.widths) - 1)
    depth = sampler.choice(depths)
    width_pair = ((depth, model.widths[w]), (depth, model.widths[w+1]))
    configs.update(depth_pair + width_pair)
    return sorted(configs), [depth_pair, width_pair]


def loss_for(outputs, configs, pairs, target, teacher, config, threshold, largest):
    if config.get("full_grid_training"):
        predictions = torch.stack([outputs[key] for key in configs])
        label = (predictions - target.unsqueeze(0)).square().mean()
        kd = (predictions - teacher.unsqueeze(0)).square().mean()
    else:
        label = torch.stack([(outputs[key] - target).square().mean() for key in configs]).mean()
        kd = torch.stack([(outputs[key] - teacher).square().mean() for key in configs]).mean()
    loss = label + config["kd_weight"] * kd
    if largest in outputs and len(configs) > 1:
        loss = loss + config["max_anchor_weight"] * (outputs[largest] - teacher).square().mean()
        loss = loss + config.get("maximum_label_weight", 0.) * (outputs[largest] - target).square().mean()
    if threshold is not None:
        for smaller, larger in pairs:
            a, b = outputs[smaller], outputs[larger]
            with torch.no_grad():
                gate = ((a - target).square().mean(1) <= threshold) & ((b - target).square().mean(1) <= threshold)
            # Empty gates contribute zero; no forced matching of inaccurate outputs.
            distance = (a - b.detach()).square().mean(1)
            loss = loss + config["consistency_weight"] * (distance * gate).sum() / gate.sum().clamp_min(1) / len(pairs)
    return loss, label


def selection_key(config, maximum_mse, grid_mse, teacher_mse, maximum_guard=None):
    if maximum_guard is not None:
        return (0 if maximum_mse <= maximum_guard else 1, grid_mse, maximum_mse)
    quality = maximum_mse <= teacher_mse * config["maximum_validation_degradation"]
    if config.get("maximum_first_selection"):
        return (0 if quality else 1, 0. if quality else maximum_mse, grid_mse)
    return (0 if quality else 1, grid_mse)


@torch.inference_mode()
def teacher_cache(config, data, output, device, batch_size):
    path = output / "teacher_predictions.npz"
    metadata = {"teacher_sha256": config["teacher_sha256"], "dataset_file_sha256": config["dataset_file_sha256"],
                "space": "normalized_output", "input_contract": config["input_contract"]}
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"].item())) != metadata:
                raise ValueError("stale teacher cache")
            result = {key: torch.from_numpy(archive[key].copy()) for key in data}
        if any(result[key].shape != data[key][1].shape or not torch.isfinite(result[key]).all() for key in data):
            raise ValueError("corrupt teacher cache")
        return result
    teacher = load_teacher(config).to(device)
    result = {}
    for split, (x, _) in data.items():
        predictions = []
        for start in range(0, len(x), batch_size):
            predictions.append(teacher(x[start:start+batch_size].to(device)).cpu())
            if start == 0 or (start // batch_size + 1) % 100 == 0:
                print(f"Teacher cache {split}: {min(start+batch_size, len(x))}/{len(x)}", flush=True)
        result[split] = torch.cat(predictions)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, metadata=json.dumps(metadata), **{key: value.numpy() for key, value in result.items()})
    temporary.replace(path)
    del teacher
    return result


@torch.inference_mode()
def calibrate_consistency(model, x, y, config):
    # Fixed TRAIN-only rows; never validation/test-derived confidence gates.
    count = min(config["consistency_calibration_samples"], len(x))
    indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(config["seed"] + 101))[:count]
    device = next(model.parameters()).device
    indices = indices.to(x.device)
    errors = []
    model.eval()
    for part in indices.split(config["evaluation_batch_size"]):
        chunk, target = x[part].to(device), y[part].to(device)
        outputs = model.outputs(model.features(chunk), grid(model))
        errors.append(torch.stack([(value - target).square().mean(1) for value in outputs.values()]).cpu())
    return float(torch.quantile(torch.cat(errors, dim=1).flatten(), config["consistency_train_quantile"]))


def continuation_state(config, output):
    source = (ROOT / config["initial_supernet_checkpoint"]).resolve()
    if output == source.parent or output.is_relative_to(source.parent):
        raise ValueError("continuation output must be outside the source run")
    if sha256(source) != config["initial_supernet_sha256"]:
        raise ValueError("continuation checkpoint hash changed")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if saved.get("network_contract") != NETWORK_CONTRACT:
        raise ValueError("continuation requires a causal supernet checkpoint")
    for key in ("input_dim", "input_contract", "group_width", "depth_choices", "width_choices",
                "teacher_sha256", "dataset_file_sha256"):
        if saved["config"][key] != config[key]:
            raise ValueError(f"continuation contract mismatch: {key}")
    return saved["model_state"]


def train(config, output, *, device="cuda", resume=False, batch_size=None):
    if ([s["name"] for s in config["stages"]] not in (["adapt", "heads", "joint"], ["heads", "joint"])
            or any(s["epochs"] < 1 for s in config["stages"])
            or not 0 < config["consistency_train_quantile"] < 1):
        raise ValueError("requires positive adapt/heads/joint or heads/joint budgets and a valid TRAIN quantile")
    if "grid_first_maximum_guard_factor" in config and (
            config["grid_first_maximum_guard_factor"] < 1 or "initial_supernet_checkpoint" not in config):
        raise ValueError("grid-first selection requires a source supernet and a guard factor >= 1")
    torch.set_num_threads(config["cpu_threads"])
    # FP32 baseline, no AMP or hidden TF32 policy changes across execution paths.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly if intended")
    output = output.resolve()
    protected = [(ROOT / config["data_dir"]).resolve(), (ROOT / config["teacher_checkpoint"]).resolve().parent]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("output must not be inside the frozen teacher or dataset directory")
    saved_config = output / "config.json"
    if saved_config.exists() and (not resume or json.loads(saved_config.read_text(encoding="utf-8")) != config):
        raise ValueError("existing run needs --resume and exactly the same config")
    if output.exists() and any(output.iterdir()) and not saved_config.exists():
        raise ValueError("output is not an initialized supernet run; choose an empty directory")
    norm, data, _ = verify_assets(config)
    if "initial_supernet_checkpoint" in config:
        initial_state = continuation_state(config, output)
    else:
        initial_state = None
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(saved_config, config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    sampler = random.Random(config["seed"] + 37)
    actual_batch = config["batch_size"] if batch_size is None else batch_size
    if actual_batch < 1:
        raise ValueError("batch size must be positive")
    evaluation_batch = min(actual_batch, config["evaluation_batch_size"])
    cached = teacher_cache(config, data, output, device, actual_batch)
    teacher_val = float((cached["validation"] - data["validation"][1]).square().mean())
    model = model_from_config(config)
    latest = output / "latest_checkpoint.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if resume and latest.exists() else None
    if saved:
        if saved["config"] != config or saved["network_contract"] != NETWORK_CONTRACT:
            raise ValueError("checkpoint configuration mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
    else:
        if "initial_supernet_checkpoint" in config:
            model.load_state_dict(initial_state, strict=True)
        else:
            teacher = load_teacher(config)
            model.initialize_from_teacher(teacher)
            del teacher
    model.to(device)
    del initial_state
    x_train, y_train = (t.to(device) for t in data["train"])
    teacher_train = cached["train"].to(device)
    x_val, y_val = data["validation"]
    completed = int(saved["epoch"]) if saved else 0
    total_epochs = sum(s["epochs"] for s in config["stages"])
    history = saved["history"] if saved else []
    best_key = tuple(saved["best_key"]) if saved else ((2, 1e30, 1e30)
        if config.get("maximum_first_selection") or "grid_first_maximum_guard_factor" in config else (2, 1e30))
    maximum_guard = saved.get("maximum_guard_mse") if saved else None
    maximum_best = saved.get("maximum_best_validation_mse", 1e30) if saved else 1e30
    adapt_best = saved["adapt_best"] if saved else 1e30
    threshold = saved["consistency_threshold"] if saved else None
    stage_index = saved["stage_index"] if saved else 0
    optimizer = optimizer_for(model, config["stages"][stage_index], config)
    if saved:
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"], sampler, device)
    runtime_events = saved.get("runtime_events", []) if saved else []
    # A shutdown may happen between saving an improved best and saving latest.
    if saved and (output / "best_model.pt").exists():
        retained = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
        best_key = min(best_key, tuple(retained["best_key"]))
        del retained
    if saved and (output / "adapt_best_model.pt").exists():
        retained = torch.load(output / "adapt_best_model.pt", map_location="cpu", weights_only=False)
        adapt_best = min(adapt_best, retained["adapt_best"])
        del retained
    if config.get("full_grid_training") and (output / "maximum_best_model.pt").exists():
        retained = torch.load(output / "maximum_best_model.pt", map_location="cpu", weights_only=False)
        maximum_best = min(maximum_best, retained["maximum_best_validation_mse"])
        del retained
    if saved and actual_batch != saved["actual_batch_size"]:
        runtime_events.append({"after_epoch": completed, "batch_from": saved["actual_batch_size"], "batch_to": actual_batch,
                               "note": "batch change breaks bit-for-bit training trajectory, not saved weights"})

    def weights_payload(epoch):
        return {"network_contract": NETWORK_CONTRACT, "config": config, "model_state": model.state_dict(),
                "epoch": epoch, "teacher_validation_mse": teacher_val, "best_key": best_key,
                "maximum_guard_mse": maximum_guard, "maximum_best_validation_mse": maximum_best,
                "consistency_threshold": threshold, "adapt_best": adapt_best}

    def save_latest(epoch):
        atomic_torch(latest, {**weights_payload(epoch), "optimizer_state": optimizer.state_dict(),
            "stage_index": stage_index, "history": history, "rng_state": rng_state(sampler),
            "adapt_best": adapt_best, "actual_batch_size": actual_batch, "runtime_events": runtime_events,
            "scheduler_state": {"clock": "completed stage epoch; cosine derived from immutable config", "completed_epoch": epoch}})

    if not saved:
        with torch.inference_mode():
            before = sum(float((model(x_val[i:i+actual_batch].to(device)) - y_val[i:i+actual_batch].to(device)).square().sum())
                         for i in range(0, len(x_val), actual_batch)) / (len(x_val) * 4)
        if config.get("maximum_first_selection") or "grid_first_maximum_guard_factor" in config:
            initial_grid, _ = evaluate_grid(model, x_val, y_val, evaluation_batch)
            initial_max = initial_grid[(model.max_depth, model.max_width)]
            if "grid_first_maximum_guard_factor" in config:
                maximum_guard = initial_max * config["grid_first_maximum_guard_factor"]
            best_key = selection_key(config, initial_max, float(np.mean(list(initial_grid.values()))),
                                     teacher_val, maximum_guard)
            maximum_best = initial_max
            atomic_torch(output / "best_model.pt", weights_payload(0))
            if config.get("full_grid_training"):
                atomic_torch(output / "maximum_best_model.pt", weights_payload(0))
        atomic_json(output / "initial_metrics.json", {"teacher_validation_mse": teacher_val,
            "masked_initial_maximum_validation_mse": before, "maximum_path_is_NOT_teacher_equivalent": True,
            "initial_supernet_checkpoint": config.get("initial_supernet_checkpoint"),
            "initial_supernet_sha256": config.get("initial_supernet_sha256"),
            "maximum_guard_mse": maximum_guard,
            "test_read": False})
        save_latest(0)
    print(f"Supernet: {len(model.depths)} depths x {len(model.widths)} widths; device={device}; "
          f"resume_epoch={completed}; batch={actual_batch}; fixed budget={total_epochs}; NO early stopping", flush=True)
    started = time.perf_counter()
    try:
        for epoch in range(completed + 1, total_epochs + 1):
            new_index, local_epoch = stage_at(config, epoch)
            if new_index != stage_index:
                if config["stages"][stage_index]["name"] == "adapt":
                    best_adapt = torch.load(output / "adapt_best_model.pt", map_location="cpu", weights_only=False)
                    model.load_state_dict(best_adapt["model_state"], strict=True)
                    del best_adapt
                stage_index = new_index
                optimizer = optimizer_for(model, config["stages"][stage_index], config)
                if config["stages"][stage_index]["name"] == "joint":
                    threshold = calibrate_consistency(model, x_train, y_train, config)
                save_latest(epoch - 1)
            stage = config["stages"][stage_index]
            factor = stage["min_lr_ratio"] + (1-stage["min_lr_ratio"]) * (1 + math.cos(math.pi*(local_epoch-1)/max(stage["epochs"]-1, 1))) / 2
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            model.train()
            order = torch.randperm(len(x_train), device=device)
            label_sum, total_sum, samples = 0., 0., 0
            for step, indices in enumerate(order.split(actual_batch), 1):
                configs, pairs = sample_configs(model, stage["name"], sampler,
                                                stage["name"] == "heads" and stage.get("maximum_head_lr", 0) > 0)
                if config.get("full_grid_training"):
                    configs = grid(model)
                optimizer.zero_grad(set_to_none=True)
                outputs = model.outputs(model.features(x_train[indices]), configs)
                loss, label = loss_for(outputs, configs, pairs, y_train[indices], teacher_train[indices], config,
                                       threshold if stage["name"] == "joint" else None, (model.max_depth, model.max_width))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss; previous complete epoch retained")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              config["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                samples += len(indices)
                total_sum += float(loss.detach()) * len(indices)
                label_sum += float(label.detach()) * len(indices)
                if step % 100 == 0:
                    print(f"epoch={epoch}/{total_epochs} stage={stage['name']} step={step}/{math.ceil(len(order)/actual_batch)} "
                          f"label={label_sum/samples:.7g}", flush=True)
            mse, _ = evaluate_grid(model, x_val, y_val, evaluation_batch)
            max_mse = mse[(model.max_depth, model.max_width)]
            grid_mse = float(np.mean(list(mse.values())))
            quality = max_mse <= teacher_val * config["maximum_validation_degradation"]
            improved = False
            if stage["name"] == "adapt" and max_mse < adapt_best:
                adapt_best = max_mse
                atomic_torch(output / "adapt_best_model.pt", weights_payload(epoch))
            if config.get("full_grid_training") and max_mse < maximum_best:
                maximum_best = max_mse
                atomic_torch(output / "maximum_best_model.pt", weights_payload(epoch))
            if stage["name"] == "joint" or config.get("maximum_first_selection") or maximum_guard is not None:
                candidate = selection_key(config, max_mse, grid_mse, teacher_val, maximum_guard)
                if candidate < best_key:
                    best_key, improved = candidate, True
                    atomic_torch(output / "best_model.pt", weights_payload(epoch))
            row = {"epoch": epoch, "stage": stage["name"], "stage_epoch": local_epoch,
                   "sampled_train_label_mse": label_sum/samples, "train_total_loss": total_sum/samples,
                   "validation_grid_mse": grid_mse, "validation_maximum_mse": max_mse,
                   "maximum_quality_goal_met": quality, "learning_rates": [g["lr"] for g in optimizer.param_groups],
                   "maximum_guard_passed": max_mse <= maximum_guard if maximum_guard is not None else None,
                   "configurations_per_batch": len(configs),
                   "elapsed_this_invocation_seconds": time.perf_counter()-started,
                   "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None,
                   "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20 if device == "cuda" else None}
            history.append(row)
            save_latest(epoch)
            atomic_json(output / "history.json", history)
            atomic_json(output / "validation_grid_latest.json", {f"d{d}_w{w}": value for (d, w), value in mse.items()})
            print(f"epoch={epoch:04d}/{total_epochs} {stage['name']}:{local_epoch} train_label={row['sampled_train_label_mse']:.7g} "
                  f"val_grid={grid_mse:.7g} val_max={max_mse:.7g} max_goal={quality} best={improved} "
                  f"GPU_peak_MiB={row['peak_cuda_allocated_mib']}", flush=True)
    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError):
        print(f"Interrupted. Keep {latest}; resume replays the unfinished epoch. "
              "For CUDA OOM only, retry with --batch-size 256; other GPU/driver errors are not repaired by this.", flush=True)
        raise
    atomic_torch(output / "last_model.pt", weights_payload(total_epochs))
    selected = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    selected_grid_mse = selected["best_key"][1] if maximum_guard is not None else selected["best_key"][-1]
    selected_maximum_mse = (next(row["validation_maximum_mse"] for row in history if row["epoch"] == selected["epoch"])
                            if selected["epoch"] else json.loads((output / "initial_metrics.json").read_text(encoding="utf-8"))[
                                "masked_initial_maximum_validation_mse"])
    atomic_json(output / "completion.json", {"epochs_completed": total_epochs, "configurations": len(grid(model)),
        "best_selection": ("uniform grid MSE subject to the initial maximum-path guard"
                           if maximum_guard is not None else
                           "quality goal first; if unmet, maximum MSE first, then grid MSE"
                           if config.get("maximum_first_selection") else
                           "maximum-path quality goal first, then uniform normalized validation grid MSE"),
        "best_checkpoint_epoch": selected["epoch"],
        "maximum_quality_goal_met": selected_maximum_mse <= teacher_val * config["maximum_validation_degradation"],
        "best_grid_validation_mse": selected_grid_mse, "best_maximum_validation_mse": selected_maximum_mse,
        "maximum_guard_mse": maximum_guard,
        "maximum_best_validation_mse": maximum_best if config.get("full_grid_training") else None,
        "teacher_validation_mse": teacher_val, "test_read": False, "runtime_events": runtime_events})
    print(f"Training complete. Shared best checkpoint: {output / 'best_model.pt'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="resume if present; otherwise initialize a new run")
    parser.add_argument("--batch-size", type=int, help="runtime override; recorded in checkpoint")
    parser.add_argument("--preflight", action="store_true", help="read-only hashes/data/source checks; no training")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.preflight:
        verify_assets(config)
        if "initial_supernet_checkpoint" in config:
            continuation_state(config, args.output_dir.resolve())
        teacher = load_teacher(config)
        print(json.dumps({"status": "PASS", "teacher_loaded": True, "input_dim": config["input_dim"],
                          "configurations": len(config["depth_choices"])*len(config["width_choices"]),
                          "epochs": sum(s["epochs"] for s in config["stages"]), "test_read": False}, indent=2))
        del teacher
        return
    train(config, args.output_dir, device=args.device, resume=args.resume, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
