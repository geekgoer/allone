"""Small shared helpers for the new, isolated experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..eval_offline import load_normalization, load_split
from ..models.residual_mlp import ResidualMLP
from .model import CausalSupernet, NETWORK_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "controller_architecture_study/configs/incremental_supernet_237d_v2_causal_width.json"
DEFAULT_RUN = ROOT / "controller_architecture_study/results/incremental_supernet_237d_v2_causal_width"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def model_from_config(config):
    return CausalSupernet(config["input_dim"], 4, config["depth_choices"], config["width_choices"],
                          config["group_width"])


def grid(model):
    return [(d, w) for d in model.depths for w in model.widths]


def load_teacher(config):
    # Only load trusted LOCAL checkpoints produced by this project.
    saved = torch.load(ROOT / config["teacher_checkpoint"], map_location="cpu", weights_only=False)
    if (saved.get("model_type") != "residual_mlp" or saved["input_dim"] != config["input_dim"]
            or saved["hidden_layers"] != max(config["depth_choices"])
            or saved["hidden_width"] != max(config["width_choices"])):
        raise ValueError("teacher checkpoint architecture mismatch")
    model = ResidualMLP(config["input_dim"], 4, saved["hidden_layers"], saved["hidden_width"])
    model.load_state_dict(saved["model_state"], strict=True)
    return model.eval().requires_grad_(False)


def verify_assets(config):
    data_dir = ROOT / config["data_dir"]
    expected = config["dataset_file_sha256"]
    hashes = {name: sha256(data_dir / name) for name in expected}
    if hashes != expected or sha256(ROOT / config["teacher_checkpoint"]) != config["teacher_sha256"]:
        raise ValueError("frozen dataset or teacher hash changed; do not reuse this experiment")
    metadata = json.loads((data_dir / "normalization.json").read_text(encoding="utf-8"))
    if metadata["input_contract"] != config["input_contract"]:
        raise ValueError("237D feature contract mismatch")
    norm = load_normalization(data_dir)
    data = {}
    families = {}
    for split in ("train", "validation"):
        x, u = load_split(data_dir, split)
        if x.shape != (config["sample_counts"][split], config["input_dim"]):
            raise ValueError(f"unexpected {split} shape: {x.shape}")
        data[split] = (torch.from_numpy((x - norm["x_mean"]) / norm["x_std"]),
                       torch.from_numpy((u - norm["u_mean"]) / norm["u_std"]))
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            families[split] = archive["family_id"].copy()
    # Deliberately do not open test.npz.
    return norm, data, families


def load_supernet(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("network_contract") != NETWORK_CONTRACT:
        raise ValueError("not a causal depth/width supernet checkpoint")
    model = model_from_config(saved["config"])
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval(), saved


@torch.inference_mode()
def evaluate_grid(model, x, target, batch_size=512, return_predictions=False):
    model.eval()
    configs = grid(model)
    sums = np.zeros(len(configs), dtype=np.float64)
    predictions = {key: np.empty((len(x), 4), dtype=np.float32) for key in configs} if return_predictions else None
    device = next(model.parameters()).device
    for start in range(0, len(x), batch_size):
        chunk = x[start:start+batch_size].to(device)
        y = target[start:start+batch_size].to(device)
        output = model.outputs(model.features(chunk), configs)
        errors = torch.stack([(output[key] - y).square().sum() for key in configs])
        sums += errors.cpu().numpy().astype(np.float64)
        if predictions is not None:
            for key in configs:
                predictions[key][start:start+len(chunk)] = output[key].cpu().numpy()
    return {key: float(value / (len(x) * 4)) for key, value in zip(configs, sums)}, predictions


def rng_state(sampler):
    import random
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "sampler": sampler.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state, sampler, device):
    import random
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    sampler.setstate(state["sampler"])
    if device.startswith("cuda") and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
