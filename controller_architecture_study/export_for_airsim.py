"""Export an M1 checkpoint for the existing gate-target AirSim controller."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .eval_offline import load_checkpoint, load_normalization, load_split


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "experiments/paper_track_p_12d_5ms_v1/dnn_dataset"


def export(checkpoint: Path, output_dir: Path, data_dir: Path = DEFAULT_DATA) -> dict:
    import torch

    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    data_dir = data_dir.resolve()
    target = output_dir / "controller_scripted.pt"
    if target.exists():
        raise FileExistsError(f"export already exists; choose a new directory: {target}")
    model = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    norm = load_normalization(data_dir)
    manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["input_dimensions"] != 12 or manifest["output_dimensions"] != 4:
        raise ValueError("dataset contract is not 12-D to 4-D")
    if manifest["input_contract"] != "[p_gate-p, r_gate-r, velocity, acceleration] in AirSim NED":
        raise ValueError("unexpected gate-input contract")
    x, _ = load_split(data_dir, "validation", limit=128)
    x_norm = torch.from_numpy((x - norm["x_mean"]) / norm["x_std"])
    scripted = torch.jit.script(model)
    with torch.inference_mode():
        reference = model(x_norm).numpy()
        scripted_output = scripted(x_norm).numpy()
    max_difference = float(np.max(np.abs(reference - scripted_output)))
    if max_difference > 1e-5:
        raise RuntimeError(f"TorchScript output mismatch: {max_difference}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted.save(str(target))
    shutil.copyfile(data_dir / "normalization.json", output_dir / "normalization.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = {
        "model_type": payload.get("model_type", "plain_mlp"),
        "hidden_layers": int(payload["hidden_layers"]),
        "hidden_width": int(payload["hidden_width"]),
        "input_dimensions": 12,
        "output_dimensions": 4,
        "input_contract": manifest["input_contract"],
        "input_features": manifest["input_features"],
        "reference_speed_values": manifest["reference_speed_values"],
        "normalization_inside_model": False,
        "normalization_file": "normalization.json",
        "checkpoint": str(checkpoint),
        "torchscript_max_abs_difference": max_difference,
    }
    (output_dir / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    # Verify the artifact through the exact runtime loader used by AirSim.
    from dnn_controller import DNNController

    runtime = DNNController(target, device="cpu")
    if runtime.input_dim != 12 or runtime.input_contract != manifest["input_contract"]:
        raise RuntimeError("AirSim controller contract mismatch")
    physical = reference[0] * norm["u_std"] + norm["u_mean"]
    if not np.allclose(runtime.solve_raw(x[0]), physical, atol=1e-5, rtol=0):
        raise RuntimeError("AirSim controller output mismatch")
    return {"model": str(target), "normalization": str(output_dir / "normalization.json"),
            "model_spec": str(output_dir / "model_spec.json"), "torchscript_max_abs_difference": max_difference,
            "airsim_raw_output_max_abs_difference": float(np.max(np.abs(runtime.solve_raw(x[0]) - physical)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output_dir, args.data_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
