"""Train matched 10x512 diagnostic controls, without reading/evaluating test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from gap_diagnosis import (ARMS, OUTPUT, ROOT, digest, error_summary, predict_physical,
                          read_npz, take, write_json)

sys.path.insert(0, str(ROOT))
from dnn_model_utils import build_model, state_dict_cpu
from prepare_dnn_dataset import normalization_from_train
from train_dnn_controller import _atomic_torch_save, _run_epoch, _seed_everything


def train_arm(torch, args, arm, data, design):
    indices = read_npz(args.experiment / f"{arm}_indices.npz")
    splits = {s: take(data, i) for s, i in indices.items()}
    norm = {k: v.astype(np.float32) for k, v in normalization_from_train(
        splits["train"]["X"], splits["train"]["U"]).items()}
    directory = args.experiment / "runs" / f"epochs_{args.epochs}_seed_{args.seed}" / arm
    directory.mkdir(parents=True, exist_ok=True)
    spec = {"design_sha256": digest(args.experiment / "design.json"),
            "runner_sha256": digest(Path(__file__)),
            "training_helpers_sha256": digest(ROOT / "train_dnn_controller.py"),
            "model_builder_sha256": digest(ROOT / "dnn_model_utils.py"),
            "arm": arm, "epochs": args.epochs, "seed": args.seed, "batch_size": 512,
            "depth": 10, "width": 512, "learning_rate": 1e-4, "gradient_clip_norm": 1.,
            "early_stopping": False, "lr_schedule": False, "test_used": False,
            "device": args.device, "torch_version": str(torch.__version__)}
    config_path = directory / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != spec:
            raise RuntimeError(f"Run configuration changed: {directory}; use a fresh experiment/seed")
        if (directory / "metrics.json").exists():
            print(f"SKIP completed: {arm}", flush=True)
            return
        if not args.resume:
            raise RuntimeError(f"Run exists: {directory}; use --resume to continue this diagnostic only")
    else:
        if any(directory.iterdir()):
            raise RuntimeError(f"Nonempty unrecognized output: {directory}")
        write_json(config_path, spec)

    _seed_everything(torch, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    tr = splits["train"]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(((tr["X"].astype(np.float32) - norm["x_mean"]) / norm["x_std"]).astype(np.float32)),
        torch.from_numpy(((tr["U"] - norm["u_mean"]) / norm["u_std"]).astype(np.float32))),
        batch_size=512, shuffle=True, generator=generator, num_workers=0, pin_memory=False)
    model = build_model(torch, depth=10, width=512, input_dim=12).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    criterion = torch.nn.MSELoss()
    start, best, history = 1, float("inf"), []
    checkpoint_path = directory / "latest_checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        torch.set_rng_state(ck["torch_rng"])
        generator.set_state(ck["loader_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    elapsed = time.perf_counter()
    for epoch in range(start, args.epochs + 1):
        online_loss = _run_epoch(model, loader, optimizer, criterion, torch.device(args.device),
                                 torch, gradient_clip_norm=1.)
        pred = predict_physical(torch, model, splits["validation"], norm, args.device)
        val_loss = float(np.mean(((pred.astype(np.float64) - splits["validation"]["U"]) / norm["u_std"]) ** 2))
        history.append({"epoch": epoch, "train_online_normalized_mse": online_loss,
                        "validation_normalized_mse": val_loss})
        payload = {"epoch": epoch, "depth": 10, "width": 512, "input_dim": 12,
                   "model_state": state_dict_cpu(model),
                   "normalization": {k: v.tolist() for k, v in norm.items()},
                   "teacher_policy": design["teacher_policy"], "input_contract": design["input_contract"]}
        if val_loss < best:
            best = val_loss
            _atomic_torch_save(torch, payload, directory / "best_model.pt")
        # Re-evaluate TRAIN in eval mode with this exact checkpoint, not the
        # moving per-batch online loss. Test remains unopened.
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_pred = predict_physical(torch, model, tr, norm, args.device)
            train_eval = float(np.mean(((train_pred.astype(np.float64) - tr["U"]) / norm["u_std"]) ** 2))
            history[-1]["train_eval_normalized_mse"] = train_eval
            print(f"{arm} epoch={epoch}/{args.epochs} train_eval={train_eval:.8f} "
                  f"validation={val_loss:.8f} best={best:.8f}", flush=True)
        _atomic_torch_save(torch, dict(payload, optimizer_state=optimizer.state_dict(),
            best=best, history=history, torch_rng=torch.get_rng_state(),
            loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else []), checkpoint_path)
        write_json(directory / "history.json", history)
    ck = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    result = {"arm": arm, "best_epoch": ck["epoch"], "epochs_completed": args.epochs,
              "test_opened": False, "session_elapsed_seconds": time.perf_counter() - elapsed,
              "metric_note": "FULL splits, SAME best checkpoint; common scale is original training u_std.",
              "normalization": {k: v.tolist() for k, v in norm.items()}}
    for split, d in splits.items():
        pred = predict_physical(torch, model, d, norm, args.device)
        result[split] = error_summary(pred, d, design["common_normalization"], design["kkt_threshold"])
        result[split]["own_normalized_mse"] = float(np.mean(((pred.astype(np.float64) - d["U"]) / norm["u_std"]) ** 2))
    write_json(directory / "metrics.json", result)
    print(f"COMPLETE {arm}: {directory / 'metrics.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=OUTPUT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true",
                        help="Explicitly allow another compute process on CUDA (not recommended)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use the training Python environment")
        # NVML lists compute contexts only; graphics applications are not a block.
        if not args.allow_busy_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                try:
                    busy = []
                    for i in range(pynvml.nvmlDeviceGetCount()):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        busy.extend(p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                                    if p.pid != os.getpid())
                finally:
                    pynvml.nvmlShutdown()
            except Exception as exc:
                raise RuntimeError("Cannot check GPU occupancy. Confirm manually before --allow-busy-gpu.") from exc
            if busy:
                raise RuntimeError(f"Another GPU compute process is active: {sorted(set(busy))}. "
                                   "Wait for current training to finish; no process was stopped.")
    design = json.loads((args.experiment / "design.json").read_text(encoding="utf-8"))
    for name, expected in design["prepared_sha256"].items():
        if digest(args.experiment / name) != expected:
            raise RuntimeError(f"Prepared data changed: {name}")
    data = read_npz(args.experiment / "pool.npz")
    for arm in args.arms:
        train_arm(torch, args, arm, data, design)
    run_root = args.experiment / "runs" / f"epochs_{args.epochs}_seed_{args.seed}"
    summary = {arm: json.loads((run_root / arm / "metrics.json").read_text())
               for arm in ARMS if (run_root / arm / "metrics.json").exists()}
    write_json(run_root / "comparison.json", summary)
    print(f"Comparison: {run_root / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
