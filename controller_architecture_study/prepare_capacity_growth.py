"""Prepare function-preserving plain-MLP growth; CPU only, no training."""

from __future__ import annotations

import hashlib
import json
import torch

from .eval_offline import load_checkpoint, load_normalization, load_split
from .models.plain_mlp import build_plain_mlp
from .run_capacity_finetune import ROOT, write_json


SOURCE = ROOT / "controller_architecture_study/results/capacity_finetune_v1/plain_mlp_h8_w256_seed0/best_model.pt"
OUTPUT = ROOT / "controller_architecture_study/results/capacity_growth_v1/initializations"


def grow_model(source, *, hidden_layers: int, hidden_width: int, seed: int = 42):
    """Net2Wider replication and identity layers after nonnegative ReLUs.

    Outgoing contributions are split unequally, with sums of exactly one in
    real arithmetic, so duplicate neurons can acquire different gradients.
    """
    old = [layer for layer in source if isinstance(layer, torch.nn.Linear)]
    depth, width = len(old) - 1, old[0].out_features
    if hidden_layers < depth or hidden_width < width:
        raise ValueError("growth cannot reduce depth or width")
    target = build_plain_mlp(torch, input_dim=old[0].in_features,
                             output_dim=old[-1].out_features,
                             hidden_layers=hidden_layers, hidden_width=hidden_width)
    new = [layer for layer in target if isinstance(layer, torch.nn.Linear)]
    generator = torch.Generator().manual_seed(seed)
    previous_map = torch.arange(old[0].in_features)
    previous_fraction = torch.ones(old[0].in_features)
    with torch.no_grad():
        for index in range(depth):
            mapping = torch.cat((torch.arange(width), torch.randint(
                width, (hidden_width - width,), generator=generator)))
            fractions = 0.75 + 0.5 * torch.rand(hidden_width, generator=generator)
            sums = torch.zeros(width).scatter_add_(0, mapping, fractions)
            fractions /= sums[mapping]
            new[index].weight.copy_(old[index].weight[mapping][:, previous_map] * previous_fraction)
            new[index].bias.copy_(old[index].bias[mapping])
            previous_map, previous_fraction = mapping, fractions
        for index in range(depth, hidden_layers):
            new[index].weight.copy_(torch.eye(hidden_width))
            new[index].bias.zero_()
        new[-1].weight.copy_(old[-1].weight[:, previous_map] * previous_fraction)
        new[-1].bias.copy_(old[-1].bias)
    return target.eval()


def main():
    torch.set_num_threads(4)
    config = json.loads((SOURCE.parent / "config.json").read_text(encoding="utf-8"))
    if (config["model_type"], config["hidden_layers"], config["hidden_width"]) != ("plain_mlp", 8, 256):
        raise ValueError("expected the completed 8x256 plain-MLP source")
    with SOURCE.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    data_dir = ROOT / config["data_dir"]
    norm = load_normalization(data_dir)
    # Validate on training inputs only; leave validation/test selection to the runner.
    x, _ = load_split(data_dir, "train", 4096)
    inputs = torch.from_numpy((x - norm["x_mean"]) / norm["x_std"])
    source = load_checkpoint(SOURCE, device="cpu")
    with torch.inference_mode():
        reference = source(inputs)
    reports = []
    for depth, width in ((8, 512), (12, 256)):
        name = f"plain_mlp_h{depth}_w{width}_seed0"
        target = OUTPUT / name
        signature = {"source_sha256": source_hash, "hidden_layers": depth,
                     "hidden_width": width, "growth_seed": 42}
        manifest = target / "growth_check.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in signature.items()):
                raise ValueError(f"source or growth settings changed: {target}")
            model = load_checkpoint(target / "best_model.pt", device="cpu")
        else:
            # An interrupted preparation can be repeated; completed initializations
            # have a manifest and are reused without overwriting their weights.
            model = grow_model(source, hidden_layers=depth, hidden_width=width)
        with torch.inference_mode():
            prediction = model(inputs)
        maximum = float((prediction - reference).abs().max())
        torch.testing.assert_close(prediction, reference, atol=2e-5, rtol=2e-5)
        report = {**signature, "samples": len(x), "normalized_output_max_abs_difference": maximum,
                  "training_started": False, "test_split_read": False}
        if not manifest.exists():
            target.mkdir(parents=True, exist_ok=True)
            grown_config = {key: value for key, value in config.items()
                            if key not in ("init_checkpoint", "source_checkpoint_sha256")}
            grown_config.update(hidden_layers=depth, hidden_width=width)
            write_json(target / "config.json", grown_config)
            temporary = target / "best_model.pt.tmp"
            torch.save({"model_state": model.state_dict(), "hidden_layers": depth,
                        "hidden_width": width, "model_type": "plain_mlp", "epoch": 0,
                        "growth_origin": str(SOURCE), **signature}, temporary)
            temporary.replace(target / "best_model.pt")
            write_json(manifest, report)
        reports.append({"model": name, "checkpoint": str(target / "best_model.pt"), **report})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
