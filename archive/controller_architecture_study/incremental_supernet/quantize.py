"""Train one W8A16 shared controller and export the integer deployment contract."""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..eval_offline import OUTPUT_NAMES, error_metrics
from .common import (ROOT, atomic_json, atomic_torch, evaluate_grid, grid, load_supernet,
                     model_from_config, restore_rng, rng_state, sha256, verify_assets)
from .integer_reference import IntegerExecutor
from .lsq_model import DYADIC_QUANT_CONTRACT, LearnedScaleSupernet
from .torch_integer import evaluate as evaluate_torch_integer
from .model import NETWORK_CONTRACT
from .qat_model import (QUANT_CONTRACT, MIXED_QUANT_CONTRACT, QuantizedSupernet, adjacency, calibrate_activations,
                        common_function_loss)


DEFAULT_CONFIG = ROOT / "controller_architecture_study/configs/incremental_supernet_237d_m2_w8a16_v1.json"


def atomic_npz(path, **arrays):
    path = Path(path)
    with path.with_name(path.name + ".tmp").open("wb") as handle:
        np.savez(handle, **arrays)
    path.with_name(path.name + ".tmp").replace(path)


def stratified_rows(families, count, seed):
    count = min(count, len(families))
    rng, groups = np.random.default_rng(seed), np.unique(families)
    chosen = []
    for i, family in enumerate(groups):
        quota = count // len(groups) + (i < count % len(groups))
        rows = np.flatnonzero(families == family)
        chosen.extend(rng.choice(rows, min(quota, len(rows)), replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(families)), chosen)
    if len(chosen) < count:
        chosen.extend(rng.choice(remaining, count-len(chosen), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def assets(config, output):
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_sha256"]:
        raise ValueError("frozen FP32 source hash changed")
    fp32, saved = load_supernet(source)
    norm, data, families = verify_assets(saved["config"])
    protected = [source.parent, (ROOT / saved["config"]["data_dir"]).resolve(),
                 (ROOT / saved["config"]["teacher_checkpoint"]).resolve().parent]
    if "initial_qat_checkpoint" in config:
        initial = (ROOT / config["initial_qat_checkpoint"]).resolve()
        if sha256(initial) != config["initial_qat_sha256"]:
            raise ValueError("QAT initialization checkpoint hash changed")
        protected.append(initial.parent)
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("M2 output must be outside the frozen source, teacher and dataset")
    return fp32, saved["config"], norm, data, families


def make_quantized(config, model, exponents, *, fold_residual=False):
    cls = LearnedScaleSupernet if config.get("learned_weight_scales", False) else QuantizedSupernet
    return cls(model, exponents, fold_residual=fold_residual, weight_margin=config["weight_margin"],
               head_weight_bits=config.get("head_weight_bits", 8))


@torch.inference_mode()
def frozen_predictions(model, data, batch_size, device):
    model.eval()
    result = {}
    for split, (x, _) in data.items():
        result[split] = torch.cat([model(x[start:start+batch_size].to(device)).cpu()
                                  for start in range(0, len(x), batch_size)])
    return result


def switch_report(predictions, reference, target, families, thresholds, model, norm):
    """Same-observation approximation differences, NOT measured flight oscillation."""
    _, pairs = adjacency(model)
    rows, total, extra_total, elements = [], 0., 0., 0
    for a, b, axis in pairs:
        delta = predictions[a] - predictions[b]
        extra_delta = delta - (reference[a] - reference[b])
        large_error = ((reference[b] - target)**2).mean(1)
        small_error = ((reference[a] - target)**2).mean(1)
        # The mask is from FROZEN FP32 predictions. QAT cannot improve the score
        # by making its large path inaccurate and thereby disabling the gate.
        mask = (large_error <= small_error) & (large_error <= thresholds[families])
        total += float(np.sum(delta[mask].astype(np.float64)**2))
        extra_total += float(np.sum(extra_delta[mask].astype(np.float64)**2))
        elements += int(mask.sum()) * 4
        physical = np.abs(delta * norm["u_std"])
        rows.append({"from": list(a), "to": list(b), "axis": axis,
                     "normalized_switch_mse": float(np.mean(delta.astype(np.float64)**2)),
                     "extra_normalized_switch_mse_vs_fp32": float(np.mean(extra_delta.astype(np.float64)**2)),
                     "fixed_confident_rows": int(mask.sum()),
                     "physical_abs_mean_per_output": dict(zip(OUTPUT_NAMES, map(float, physical.mean(0)))),
                     "physical_abs_p99_per_output": dict(zip(OUTPUT_NAMES, map(float, np.percentile(physical, 99, axis=0))))})
    stacked = np.stack(list(predictions.values()))
    # Mean over all unordered configuration pairs, calculated without C^2 storage.
    c = len(predictions)
    pair_mse = float(stacked.astype(np.float64).var(0).mean() * 2*c/max(c-1, 1))
    return {"metric": "same_input_switch_difference_not_temporal_oscillation",
            "does_not_prove_stability": True, "reference_mask": "frozen_FP32_label_informed_not_online_oracle",
            "reliable_switch_mse": total/elements if elements else None,
            "extra_reliable_switch_mse_vs_fp32": extra_total/elements if elements else None,
            "fixed_confident_elements": elements, "all_pairs_normalized_switch_mse": pair_mse,
            "adjacent_pairs": rows}


def quality_key(metrics, switching, baseline, config):
    maximum_ok = metrics["maximum_mse"] <= baseline["maximum_mse"] * config["maximum_degradation"]
    grid_ok = metrics["grid_mse"] <= baseline["grid_mse"] * config["grid_degradation"]
    initial = baseline["switching"]["reliable_switch_mse"]
    score = switching["reliable_switch_mse"]
    switch_ok = initial is not None and score is not None and score <= max(initial, 1e-12) * config["switch_degradation"]
    rank = 0 if maximum_ok and grid_ok and switch_ok else 1 if maximum_ok and grid_ok else 2
    key = (rank, metrics["grid_mse"] + config["selection_switch_weight"] * (score or 0.), metrics["maximum_mse"])
    return key, {"maximum_quality_goal_met": maximum_ok, "grid_quality_goal_met": grid_ok,
                 "switch_non_degradation_goal_met": switch_ok, "all_quality_goals_met": rank == 0}


def metric_summary(mse, model):
    return {"maximum_mse": mse[(model.max_depth, model.max_width)],
            "grid_mse": float(np.mean(list(mse.values()))),
            "configuration_mse": {f"d{d}_w{w}": value for (d, w), value in mse.items()}}


def training(config, output, *, device="cuda", resume=False, batch_size=None):
    if config["epochs"] < 1 or config["calibration_samples"] < 1 or not 0 < config["confidence_quantile"] < 1:
        raise ValueError("invalid QAT budget or TRAIN confidence calibration")
    learned = config.get("learned_weight_scales", False)
    if learned and (not config.get("reference_preservation") or not 0 < config["scale_learning_epochs"] < config["epochs"]):
        raise ValueError("learned-scale QAT needs FP32 references and a final frozen-scale phase")
    output = output.resolve()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose CPU explicitly if intended")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fp32, source_config, norm, data, families = assets(config, output)
    marker = output / "config.json"
    if marker.exists():
        if not resume or json.loads(marker.read_text(encoding="utf-8")) != config:
            raise ValueError("existing M2 run needs --resume and the same immutable config")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("choose an empty M2 output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, config)
    actual_batch = batch_size or config["batch_size"]
    if actual_batch < 1:
        raise ValueError("batch size must be positive")
    eval_batch = min(actual_batch, config["evaluation_batch_size"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    sampler = random.Random(config["seed"])
    cache_path = output / "frozen_reference.npz"
    fp32.to(device)
    preserve_reference = config.get("reference_preservation", False)
    fp32.eval().requires_grad_(False)
    if not cache_path.exists():
        teacher = frozen_predictions(fp32, data, eval_batch, device)
        _, reference = evaluate_grid(fp32, *data["validation"], eval_batch, return_predictions=True)
        indices = stratified_rows(families["train"], config["calibration_samples"], config["seed"]+71)
        exponents, maxima = calibrate_activations(fp32, data["train"][0][indices], eval_batch, config["activation_margin"])
        errors = (teacher["train"] - data["train"][1]).square().mean(1).numpy()
        thresholds = np.zeros(int(max(families["train"]))+1, dtype=np.float32)
        for family in np.unique(families["train"]):
            thresholds[int(family)] = max(float(np.quantile(errors[families["train"] == family], config["confidence_quantile"])), 1e-8)
        configs = grid(fp32)
        atomic_npz(cache_path, metadata=json.dumps({"source_sha256": config["source_sha256"], "test_read": False}),
                   train_teacher=teacher["train"].numpy(), validation_teacher=teacher["validation"].numpy(),
                   configurations=np.asarray(configs), validation_reference=np.stack([reference[k] for k in configs]),
                   calibration_row_indices=indices, activation_exponents=exponents.numpy(),
                   activation_maxima=maxima.numpy(), family_thresholds=thresholds)
    with np.load(cache_path, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata"].item()))["source_sha256"] != config["source_sha256"]:
            raise ValueError("stale calibration/reference cache")
        teacher_train = torch.from_numpy(archive["train_teacher"].copy()).to(device)
        reference = {tuple(map(int, key)): value.copy() for key, value in
                     zip(archive["configurations"], archive["validation_reference"])}
        exponents = archive["activation_exponents"].copy()
        thresholds = archive["family_thresholds"].copy()
        indices = archive["calibration_row_indices"].copy()
    target_val = data["validation"][1].numpy()
    source_mse = {key: float(np.mean((value-target_val).astype(np.float64)**2)) for key, value in reference.items()}
    baseline = metric_summary(source_mse, fp32)
    baseline["switching"] = switch_report(reference, reference, target_val, families["validation"], thresholds, fp32, norm)
    latest = output / "latest_checkpoint.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() and resume else None
    if saved:
        model = make_quantized(config, model_from_config(source_config), exponents)
        if saved["quantization_contract"] != model.quantization_contract:
            raise ValueError("checkpoint quantization contract changed")
        model.load_state_dict(saved["model_state"], strict=True)
        model.to(device)
    else:
        initial = copy.deepcopy(fp32) if preserve_reference else fp32
        fold = True
        if "initial_qat_checkpoint" in config:
            warm = torch.load(ROOT/config["initial_qat_checkpoint"], map_location="cpu", weights_only=False)
            if (warm["config"]["source_sha256"] != config["source_sha256"] or warm["source_config"] != source_config
                    or warm["quantization_contract"] != MIXED_QUANT_CONTRACT
                    or not np.array_equal(warm["activation_exponents"], exponents)):
                raise ValueError("QAT initialization has incompatible FP32 source, model or A16 encoding")
            initial.load_state_dict({name[6:]: value for name, value in warm["model_state"].items()
                                     if name.startswith("model.")}, strict=True)
            fold = False  # The initialization checkpoint already folded alpha.
            del warm
        model = make_quantized(config, initial, exponents, fold_residual=fold).to(device)
    if not preserve_reference:
        del fp32
    model.requires_grad_(True)
    backbone = list(model.model.stem.parameters()) + list(model.model.blocks.parameters()) + list(model.model.heads[-1].parameters())
    other_heads = [p for head in model.model.heads[:-1] for p in head.parameters()]
    groups = [{"params": backbone, "lr": config["backbone_lr"], "base_lr": config["backbone_lr"]},
              {"params": other_heads, "lr": config["head_lr"], "base_lr": config["head_lr"]}]
    if learned:
        groups.append({"params": list(model.log_steps.parameters()), "lr": config["scale_lr"],
                       "base_lr": config["scale_lr"], "weight_decay": 0.})
    optimizer = torch.optim.Adam(groups, weight_decay=config["weight_decay"], foreach=False)
    x, y = (value.to(device) for value in data["train"])
    family_tensor = torch.as_tensor(families["train"], dtype=torch.long, device=device)
    threshold_tensor = torch.from_numpy(thresholds).to(device)
    history, completed, events = [], 0, []
    best_key = (3, 1e30, 1e30)
    if saved:
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"], sampler, device)
        history, completed, best_key = saved["history"], saved["epoch"], tuple(saved["best_key"])
        events = saved["runtime_events"]
        if actual_batch != saved["actual_batch_size"]:
            events.append({"after_epoch": completed, "batch_size_from": saved["actual_batch_size"],
                           "batch_size_to": actual_batch, "bit_exact_training_continuation": False})
        if (output / "best_model.pt").exists():
            retained = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
            best_key = min(best_key, tuple(retained["best_key"]))
            del retained

    def payload(epoch, full=False):
        value = {"network_contract": NETWORK_CONTRACT, "quantization_contract": model.quantization_contract,
                 "config": config, "source_config": source_config, "model_state": model.state_dict(),
                 "activation_exponents": model.activation_exponents.cpu().tolist(), "epoch": epoch,
                 "best_key": best_key, "baseline": baseline, "family_thresholds": thresholds,
                 "calibration_row_indices": indices, "test_read": False}
        if full:
            value.update(optimizer_state=optimizer.state_dict(), rng_state=rng_state(sampler),
                         history=history, actual_batch_size=actual_batch, runtime_events=events)
        return value

    def assess():
        mse, predictions = evaluate_grid(model, *data["validation"], eval_batch, return_predictions=True)
        metrics = metric_summary(mse, model)
        switching = switch_report(predictions, reference, target_val, families["validation"], thresholds, model, norm)
        key, goals = quality_key(metrics, switching, baseline, config)
        return key, metrics, switching, goals

    if not saved:
        if learned:
            model.freeze_scales()
        best_key, initial, switching, goals = assess()
        atomic_json(output / "initial_metrics.json", {"FP32": baseline, "PTQ_fake_quant": initial,
                    "PTQ_switching": switching, "goals": goals, "test_read": False})
        atomic_torch(output / "best_model.pt", payload(0))
        if learned:
            model.thaw_scales()
        atomic_torch(latest, payload(0, True))
    objective = "per-configuration FP32 preservation + switch-delta preservation" if preserve_reference else "gated consistency"
    print(f"W8/headW{model.head_weight_bits}/A16 QAT: {len(grid(model))} configs, fixed {config['epochs']} epochs, resume={completed}, "
          f"batch={actual_batch}; MPC labels + {objective}", flush=True)
    started = time.perf_counter()
    try:
        for epoch in range(completed+1, config["epochs"]+1):
            if learned and epoch > config["scale_learning_epochs"]:
                model.freeze_scales()
            stage = "learn_scales" if learned and not bool(model.scales_frozen) else "fixed_scales"
            factor = config["min_lr_ratio"] + (1-config["min_lr_ratio"])*(1+math.cos(math.pi*(epoch-1)/max(config["epochs"]-1, 1)))/2
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            model.train()
            order = torch.randperm(len(x), device=device)
            sums, count = {}, 0
            for step, part in enumerate(order.split(actual_batch), 1):
                optimizer.zero_grad(set_to_none=True)
                reference_outputs = None
                if preserve_reference:
                    with torch.no_grad():
                        reference_outputs = fp32.outputs(fp32.features(x[part]), grid(model))
                outputs = model.outputs(model.features(x[part]), grid(model))
                loss, terms = common_function_loss(outputs, y[part], teacher_train[part],
                                                    threshold_tensor, family_tensor[part], model, config, reference_outputs)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite QAT loss; last complete epoch is retained")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                model.model.enforce_mask()
                count += len(part)
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.) + float(value.detach()) * len(part)
                if step % 100 == 0:
                    print(f"QAT epoch={epoch}/{config['epochs']} step={step}/{math.ceil(len(x)/actual_batch)} "
                          f"stage={stage} label={sums['label']/count:.7g} consistency={sums['consistency']/count:.7g}", flush=True)
            key, metrics, switching, goals = assess()
            eligible = not learned or bool(model.scales_frozen)
            improved = eligible and key < best_key
            if improved:
                best_key = key
                atomic_torch(output / "best_model.pt", payload(epoch))
            row = {"epoch": epoch, **{name: value/count for name, value in sums.items()},
                   "validation_grid_mse": metrics["grid_mse"], "validation_maximum_mse": metrics["maximum_mse"],
                   "validation_reliable_switch_mse": switching["reliable_switch_mse"], **goals,
                   "validation_extra_switch_mse_vs_fp32": switching["extra_reliable_switch_mse_vs_fp32"],
                   "learning_rates": [group["lr"] for group in optimizer.param_groups],
                   "elapsed_this_invocation_seconds": time.perf_counter()-started,
                   "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None}
            if learned:
                row.update(scales_frozen=bool(model.scales_frozen), eligible_for_export=eligible)
            history.append(row)
            atomic_torch(latest, payload(epoch, True))
            atomic_json(output / "history.json", history)
            atomic_json(output / "validation_latest.json", {**metrics, "switching": switching, **goals, "test_read": False})
            print(f"epoch={epoch:04d}/{config['epochs']} label={row['label']:.7g} "
                  f"val_grid={metrics['grid_mse']:.7g} val_max={metrics['maximum_mse']:.7g} "
                  f"switch={switching['reliable_switch_mse']} quality={goals['all_quality_goals_met']} best={improved}", flush=True)
    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError):
        print(f"Interrupted; {latest} retained. Resume replays the unfinished epoch, not the whole run.", flush=True)
        raise
    atomic_torch(output / "last_model.pt", payload(config["epochs"]))
    selected = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    atomic_json(output / "completion.json", {"epochs_completed": config["epochs"], "best_epoch": selected["epoch"],
                "configurations": len(grid(model)), "selection_key": list(selected["best_key"]),
                "selection_space": "fake_quant_validation; integer quality checked during export",
                "source_sha256": config["source_sha256"], "test_read": False, "runtime_events": events})
    return output


def packed_arrays(model):
    dyadic = model.quantization_contract == DYADIC_QUANT_CONTRACT
    if dyadic and not bool(model.scales_frozen):
        raise ValueError("freeze learned scales before integer packing")
    arrays = {"activation_exponents": model.activation_exponents.cpu().numpy()}
    g = model.group_width
    clipped, weights = 0, 0
    for name, layer in model.layers():
        exponent = getattr(model, name + "_weight_exponent").cpu().numpy().astype(np.int16)
        weight = layer.weight.detach().cpu().numpy().astype(np.float64)
        if name.startswith("layer"):
            weight *= model.model.causal_mask.cpu().numpy()
        if not np.isfinite(weight).all():
            raise ValueError(f"non-finite {name} weight")
        if dyadic:
            multiplier = getattr(model, name + "_weight_multiplier").cpu().numpy().astype(np.uint16)
            arrays[name + "_weight_multiplier"] = multiplier
            scale = np.ldexp(multiplier.astype(np.float64), exponent.astype(np.int32))
            q = np.rint(weight / scale[:, None])
        else:
            q = np.rint(np.ldexp(weight, -exponent.astype(np.int32)[:, None]))
        limit = model.weight_limit(name)
        clipped += int(np.count_nonzero(np.abs(q) > limit))
        weights += int(np.count_nonzero(model.model.causal_mask.cpu().numpy())) if name.startswith("layer") else q.size
        q = q.clip(-limit, limit).astype(np.int16 if limit > 127 else np.int8)
        d = int(name[5:]) if name.startswith("layer") else int(name[4:]) if name.startswith("head") else 1
        input_exp = int(arrays["activation_exponents"][d if name.startswith("head") else d-1])
        bias = layer.bias.detach().cpu().numpy().astype(np.float64)
        bias_float = np.rint(bias / np.ldexp(scale, input_exp)) if dyadic else np.rint(np.ldexp(bias, -(exponent.astype(np.int32)+input_exp)))
        if not np.isfinite(bias_float).all() or np.any(np.abs(bias_float) > 2**47-1):
            raise OverflowError(f"{name} bias exceeds the frozen 48-bit contract")
        arrays[name + "_bias"] = bias_float.astype(np.int64)
        arrays[name + "_weight_exponent"] = exponent
        if name.startswith("layer"):
            for out in range(len(model.widths)):
                for src in range(out+1):
                    arrays[f"{name}_g{out}_from{src}"] = q[out*g:(out+1)*g, src*g:(src+1)*g]
        else:
            arrays[name + "_weight"] = q
    return arrays, {"clipped_weights": clipped, "effective_weight_values": weights}


def write_binary(path, arrays):
    layout = {}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        for name, value in arrays.items():
            padding = (-handle.tell()) % 64
            handle.write(bytes(padding))
            value = np.asarray(value, dtype=value.dtype.newbyteorder("<"), order="C")
            layout[name] = {"offset_bytes": handle.tell(), "bytes": value.nbytes,
                            "dtype": value.dtype.str, "shape": list(value.shape)}
            handle.write(value.tobytes(order="C"))
    temporary.replace(path)
    return layout


def integer_metrics(executor, x, y, batch_size):
    keys = [(d, w) for d in executor.depths for w in executor.widths]
    result = {key: np.empty((len(x), 4), dtype=np.float32) for key in keys}
    for start in range(0, len(x), batch_size):
        output = executor.all_outputs(x[start:start+batch_size])
        for key in keys:
            result[key][start:start+batch_size] = output[key]
        print(f"Integer validation: {min(start+batch_size, len(x))}/{len(x)}; all {len(keys)} configs", flush=True)
    mse = {key: float(np.mean((value-y).astype(np.float64)**2)) for key, value in result.items()}
    return mse, result


def export(checkpoint, output, *, device="cpu", golden_count=32, integer_batch_size=32):
    checkpoint, output = checkpoint.resolve(), output.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("quantization_contract") not in (QUANT_CONTRACT, MIXED_QUANT_CONTRACT, DYADIC_QUANT_CONTRACT):
        raise ValueError("not a supported integer QAT checkpoint")
    if golden_count < 1 or integer_batch_size < 1:
        raise ValueError("positive export sample/batch sizes required")
    config, source_config = saved["config"], saved["source_config"]
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    source, actual_source_config, norm, data, families = assets(config, output)
    if actual_source_config != source_config:
        raise ValueError("source contract changed")
    del source
    model = make_quantized(config, model_from_config(source_config), saved["activation_exponents"])
    if model.quantization_contract != saved["quantization_contract"]:
        raise ValueError("checkpoint head precision disagrees with quantization contract")
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval().to(device)
    dyadic = model.quantization_contract == DYADIC_QUANT_CONTRACT
    if dyadic and not bool(model.scales_frozen):
        raise ValueError("export requires a frozen-scale checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(checkpoint)
    marker = output / "export_report.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("export belongs to another checkpoint; choose a new directory")
    arrays, weight_stats = packed_arrays(model)
    weights_name = "weights_mixed" if model.head_weight_bits == 16 else "weights_int8"
    archive_name, binary_name = weights_name + ".npz", weights_name + ".bin"
    atomic_npz(output / archive_name, **arrays)
    metadata = json.loads((ROOT/source_config["data_dir"]/"normalization.json").read_text(encoding="utf-8"))
    atomic_json(output / "normalization.json", metadata)
    layout = write_binary(output / binary_name, arrays)
    calibration_rows = np.asarray(saved["calibration_row_indices"], dtype=np.int64)
    atomic_json(output / "calibration_manifest.json", {"split": "train", "test_read": False,
                "train_sha256": source_config["dataset_file_sha256"]["train.npz"],
                "train_row_indices": calibration_rows.tolist(),
                "family_counts": {str(int(f)): int(np.count_nonzero(families["train"][calibration_rows] == f))
                                  for f in np.unique(families["train"][calibration_rows])},
                "confidence_thresholds_from_train": np.asarray(saved["family_thresholds"]).tolist(),
                "activation_exponents": saved["activation_exponents"], "scales_frozen_during_QAT": not dyadic,
                "activation_scales_frozen_during_QAT": True,
                "weight_scale_learning_epochs": config.get("scale_learning_epochs", 0)})
    file_hashes = {name: sha256(output/name) for name in (archive_name, binary_name, "normalization.json",
                                                       "calibration_manifest.json")}
    spec = {"network_contract": NETWORK_CONTRACT, "quantization_contract": model.quantization_contract,
            "input_contract": source_config["input_contract"], "input_dim": model.input_dim, "output_dim": 4,
            "depth_choices": model.depths, "width_choices": model.widths, "group_width": model.group_width,
            "configuration_count": len(grid(model)), "package_version": checkpoint_hash,
            "source_fp32_sha256": config["source_sha256"], "checkpoint_epoch": saved["epoch"],
            "weights": f"backbone symmetric INT8; heads symmetric INT{model.head_weight_bits}; per-output-channel power-of-two scales",
            "backbone_weight_bits": 8, "head_weight_bits": model.head_weight_bits,
            "weights_archive": archive_name, "weights_binary": binary_name,
            "activations": "symmetric INT16 [-32767,32767], frozen per-layer power-of-two scales",
            "scale_definition": "real_value = integer_value * 2**exponent",
            "mac_and_bias": "signed 48-bit; stored biases are little-endian INT64 with range checked",
            "alignment_intermediates": "INT64; checked working magnitude <= 2**62-1 before left shift",
            "rounding": "nearest ties-to-even; negative right shifts use floor plus remainder correction",
            "residual": "alpha folded into branch W/b before QAT; align before addition; ReLU, round, saturate ONCE",
            "original_residual_scale": model.original_residual_scale,
            "head": "bias once; sum integer group contributions in wide accumulator; no A8/A16 output saturation",
            "normalization": "PS float32 raw input normalization ONCE; physical output conversion ONCE",
            "incremental_context": "same sample + package_version; cache all completed layers/groups",
            "pause_boundary": "complete output group after residual/ReLU/requantization",
            "causal_dependency": "output group g reads only input groups 0..g",
            "array_layout": layout, "package_file_sha256": file_hashes, "test_read": False}
    if dyadic:
        spec.update(weights="W8 backbone / W16 heads; shared learned row scales M*2**exponent",
                    scale_definition="weight_real = code * uint16_multiplier * 2**weight_base_exponent; activation_real = code * 2**activation_exponent",
                    multiplier_bits=16, scales_frozen_for_deployment=True,
                    alignment_intermediates="signed 63-bit working range; check MAC * M and residual alignment BEFORE arithmetic",
                    residual="48-bit branch MAC+bias; multiply by M; align with identity; add, ReLU and round ONCE",
                    head="sum raw 48-bit group MACs plus bias ONCE, then multiply total by M; output codes contain M",
                    integer_bounds_verified_for_all_A16_inputs=False,
                    learning_method="LSQ-style row step gradients, 16-bit dyadic snapping throughout QAT, final frozen-scale phase")
    atomic_json(output / "quantization_spec.json", spec)
    executor = IntegerExecutor.load(output)
    if dyadic:
        spec["integer_bounds_verified_for_all_A16_inputs"] = True
        atomic_json(output / "quantization_spec.json", spec)
    x, y = (value.numpy() for value in data["validation"])
    mse, predictions = integer_metrics(executor, x, y, integer_batch_size)
    with np.load(checkpoint.parent/"frozen_reference.npz", allow_pickle=False) as archive:
        if json.loads(str(archive["metadata"].item()))["source_sha256"] != config["source_sha256"]:
            raise ValueError("reference cache does not match frozen source")
        reference = {tuple(map(int, k)): v.copy() for k, v in zip(archive["configurations"], archive["validation_reference"])}
    switching = switch_report(predictions, reference, y, families["validation"], saved["family_thresholds"], model, norm)
    metrics = metric_summary(mse, model)
    _, goals = quality_key(metrics, switching, saved["baseline"], config)
    physical_targets = y * norm["u_std"] + norm["u_mean"]
    rows = []
    for (d, w), value in predictions.items():
        row = {"depth": d, "width": w, **error_metrics(value, physical_targets, norm)}
        row["by_family_id"] = {str(int(f)): error_metrics(value[families["validation"] == f],
                                 physical_targets[families["validation"] == f], norm) for f in np.unique(families["validation"])}
        rows.append(row)
    atomic_json(output / "offline_metrics.json", {"space": "true_integer_validation_not_fake_quant", "test_read": False,
                **metrics, **goals, "configurations": rows})
    atomic_json(output / "switch_consistency_report.json", {"FP32": saved["baseline"]["switching"], "integer": switching,
                "temporal_flight_oscillation_verified": False, "independent_model_pool_comparison_done": False,
                "no_contiguous_pairs_in_current_dataset": True, "test_read": False})
    # Complete integer outputs for every configuration, plus exact wide head codes.
    indices = np.linspace(0, len(x)-1, min(golden_count, len(x)), dtype=int)
    raw = x[indices] * norm["x_std"] + norm["x_mean"]
    q_input = executor.encode_raw(raw, norm)
    context = executor.context(q_input, "golden")
    context.grow(model.max_depth, model.max_width, sample_id="golden", model_version=executor.version)
    keys = grid(model)
    normalized, accumulators, output_exponents = [], [], []
    for d, w in keys:
        values, codes = context.readout(d, w)
        normalized.append(values)
        accumulators.append(codes)
        output_exponents.append(int(arrays["activation_exponents"][d])+arrays[f"head{d}_weight_exponent"])
    normalized = np.asarray(normalized)
    gold_physical = normalized * norm["u_std"] + norm["u_mean"]
    reencoded = (raw - norm["x_mean"]) / norm["x_std"]
    with torch.inference_mode():
        fake = model.outputs(model.features(torch.from_numpy(reencoded).to(device)), keys)
        fake = np.stack([fake[k].cpu().numpy() for k in keys])
    fake_error = float(np.max(np.abs((fake-normalized) * norm["u_std"])))
    qat_float32, qat_float32_error = fake, fake_error
    strict_error, strict_bit_exact = None, None
    strict_outputs, strict_accumulators = None, None
    if dyadic:
        checked, checked_codes = evaluate_torch_integer(model, torch.from_numpy(reencoded).to(device), keys)
        strict_outputs = np.stack([checked[k].cpu().numpy() for k in keys])
        strict_accumulators = np.stack([checked_codes[k].cpu().numpy() for k in keys])
        strict_bit_exact = np.array_equal(strict_accumulators, np.asarray(accumulators))
        strict_error = float(np.max(np.abs((strict_outputs-normalized) * norm["u_std"])))
        if not strict_bit_exact or strict_error != 0.:
            raise RuntimeError(f"independent Torch/NumPy integer mismatch: bit_exact={strict_bit_exact}, physical_error={strict_error}")
        # FP32 accumulation and FP32 weight/scale division are a training
        # surrogate. Near RNE boundaries they can change codes. Check the same
        # frozen fake-quant function in FP64, and retain the FP32 discrepancy.
        model.double()
        with torch.inference_mode():
            precise = model.outputs(model.features(torch.from_numpy(reencoded).to(device=device, dtype=torch.float64)), keys)
            fake = np.stack([precise[k].cpu().numpy() for k in keys])
        model.float()
        fake_error = float(np.max(np.abs((fake-normalized) * norm["u_std"])))
        print(f"Golden numerics: FP32 QAT surrogate={qat_float32_error:.9g}, "
              f"FP64 emulation={fake_error:.9g}, independent integer bit_exact={strict_bit_exact}", flush=True)
    if fake_error > config["fake_integer_physical_tolerance"]:
        raise RuntimeError(f"fake-quant/integer discrepancy {fake_error} exceeds the documented tolerance")
    orders = [[(model.max_depth, model.widths[0]), (model.max_depth, model.max_width)],
              [(model.depths[0], model.max_width), (model.max_depth, model.max_width)]]
    for order in orders:
        continued = executor.context(q_input[:2], "continuation")
        first = True
        old = {}
        for d, w in order:
            (_, codes), _ = continued.grow(d, w, sample_id="continuation", model_version=executor.version)
            if first:
                old = {key: value.copy() for key, value in continued.activations.items()}
                first = False
        if not np.array_equal(codes, context.readout(model.max_depth, model.max_width)[1][:2]):
            raise RuntimeError("integer incremental output differs from direct execution")
        if any(not np.array_equal(value, continued.activations[key]) for key, value in old.items()):
            raise RuntimeError("growth changed an already-computed causal feature")
    extra_golden = {} if not dyadic else {
        "outputs_torch_integer_physical": strict_outputs*norm["u_std"]+norm["u_mean"],
        "outputs_torch_accumulator_int64": strict_accumulators,
        "outputs_qat_float32_physical": qat_float32*norm["u_std"]+norm["u_mean"]}
    atomic_npz(output / "golden_vectors_int.npz", X_raw=raw, X_normalized=reencoded, X_int16=q_input,
               configurations=np.asarray(keys), outputs_accumulator_int64=np.asarray(accumulators),
               output_exponents=np.asarray(output_exponents), outputs_normalized=normalized,
               outputs_physical=gold_physical, outputs_fake_quant_physical=fake*norm["u_std"]+norm["u_mean"],
               validation_row_indices=indices, **extra_golden)
    report = {"status": "PASS" if goals["all_quality_goals_met"] else "NUMERICS_PASS_QUALITY_NOT_MET",
              "quantization_contract": model.quantization_contract, "head_weight_bits": model.head_weight_bits,
              "deployment_scales_frozen": bool(model.scales_frozen) if dyadic else True,
              "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": saved["epoch"], "configurations": len(keys),
              "integer_direct_incremental_bit_exact": True, "cached_prefix_unchanged": True,
              "incremental_checked_samples": min(2, len(indices)), "incremental_checked_orders": len(orders),
              "golden_samples": len(indices),
              "integer_validation_rows": len(x), "fake_integer_golden_max_abs_physical_error": fake_error,
              "numerical_acceptance": "FP64_fake_quant_and_independent_torch_vs_numpy_integer" if dyadic else "fake_quant_vs_integer",
              "fake_quant_emulation_dtype": "float64" if dyadic else "float32",
              "torch_integer_bit_exact": strict_bit_exact,
              "torch_integer_golden_max_abs_physical_error": strict_error,
              "qat_float32_golden_max_abs_physical_error": qat_float32_error,
              "qat_surrogate_within_tolerance": qat_float32_error <= config["fake_integer_physical_tolerance"],
              "activation_clipped_fraction": executor.clipped/max(executor.activation_values, 1),
              **weight_stats, **goals, "maximum_validation_mse": metrics["maximum_mse"],
              "grid_validation_mse": metrics["grid_mse"], "same_input_switch_mse": switching["reliable_switch_mse"],
              "extra_switch_mse_vs_fp32": switching["extra_reliable_switch_mse_vs_fp32"],
              "FPGA_verified": False, "flight_oscillation_reduction_verified": False, "test_read": False,
              "artifact_sha256": {name: sha256(output/name) for name in (archive_name, binary_name,
                  "normalization.json", "calibration_manifest.json", "quantization_spec.json", "offline_metrics.json",
                  "switch_consistency_report.json", "golden_vectors_int.npz")}}
    atomic_json(marker, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "export"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--golden-count", type=int, default=32)
    parser.add_argument("--integer-batch-size", type=int, default=32)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = (ROOT/config["output_dir"]).resolve()
    if args.action == "preflight":
        model, _, _, data, _ = assets(config, args.output_dir or run)
        print(json.dumps({"status": "PASS", "configurations": len(grid(model)), "epochs": config["epochs"],
                          "train_rows": len(data["train"][0]), "validation_rows": len(data["validation"][0]),
                          "source_sha256": config["source_sha256"], "test_read": False}, indent=2))
    elif args.action == "train":
        training(config, args.output_dir or run, device=args.device, resume=args.resume, batch_size=args.batch_size)
    else:
        report = export(args.checkpoint or run/"best_model.pt", args.output_dir or run/"export", device=args.device,
                        golden_count=args.golden_count, integer_batch_size=args.integer_batch_size)
        if report["status"] != "PASS":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
