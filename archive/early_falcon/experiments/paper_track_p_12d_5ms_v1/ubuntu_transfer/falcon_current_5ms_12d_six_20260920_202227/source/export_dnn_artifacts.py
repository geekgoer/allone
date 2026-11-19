"""Export a trained model to Vitis-AI-neutral weights and golden vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dnn_model_utils import build_model, linear_layers, load_normalization, model_spec, normalized_input, numpy_forward, state_dict_cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dnn_dataset_full_route_v2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--golden-count", type=int, default=1000)
    args = parser.parse_args()
    import torch

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    depth, width = int(checkpoint["depth"]), int(checkpoint["width"])
    input_dim = int(checkpoint.get("input_dim", 12))
    model = build_model(torch, depth=depth, width=width, input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalization = load_normalization(
        args.data_dir / "normalization.json",
        expected_input_dim=input_dim,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights: dict[str, np.ndarray] = {}
    layers = linear_layers(model)
    for index, layer in enumerate(layers):
        weights[f"linear_{index:02d}_weight"] = layer.weight.detach().numpy().astype(np.float32)
        weights[f"linear_{index:02d}_bias"] = layer.bias.detach().numpy().astype(np.float32)
    np.savez_compressed(args.output_dir / "weights_fp32.npz", **weights)
    (args.output_dir / "normalization.json").write_text(json.dumps({key: value.tolist() for key, value in normalization.items()}, indent=2) + "\n", encoding="utf-8")
    spec = model_spec(
        depth=depth,
        width=width,
        input_dim=input_dim,
        input_features=checkpoint.get("input_features"),
        teacher_policy=checkpoint.get("teacher_policy"),
        input_contract=checkpoint.get("input_contract"),
        normalization_path="normalization.json",
    )
    spec["reference_speed_values"] = checkpoint.get("reference_speed_values", [])
    spec["layers"] = [{"index": i, "type": "Linear", "in_features": int(layer.in_features), "out_features": int(layer.out_features), "activation_after": i < depth - 1} for i, layer in enumerate(layers)]
    (args.output_dir / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    with np.load(args.data_dir / "test.npz", allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        u = np.asarray(archive["U"], dtype=np.float32)
    count = min(args.golden_count, len(x))
    golden_x = x[:count]
    golden_norm = normalized_input(golden_x, normalization)
    with torch.no_grad():
        torch_norm = model(torch.from_numpy(golden_norm)).numpy()
    np_norm = numpy_forward(weights, golden_norm, depth=depth)
    max_error = float(np.max(np.abs(torch_norm - np_norm)))
    # The NumPy and PyTorch GEMM kernels can accumulate float32 values in a
    # different order.  The exported graph must still agree within a small
    # absolute float32 tolerance, rather than requiring bitwise identity.
    if not np.allclose(torch_norm, np_norm, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"NumPy/PyTorch export parity failed; max error={max_error}")
    u_pred = np_norm * normalization["u_std"] + normalization["u_mean"]
    np.savez_compressed(args.output_dir / "golden_vectors.npz", X_physical=golden_x, X_normalized=golden_norm, U_true=u[:count], U_pytorch=u_pred)
    manifest = {"model": str(args.model.resolve()), "input_dimensions": input_dim, "input_features": spec["input_features"], "teacher_policy": spec["teacher_policy"], "input_contract": spec.get("input_contract"), "reference_speed_values": spec.get("reference_speed_values", []), "weights": "weights_fp32.npz", "model_spec": "model_spec.json", "normalization": "normalization.json", "golden_vectors": "golden_vectors.npz", "golden_count": count, "numpy_pytorch_max_abs_error": max_error, "state_dict_keys": sorted(state_dict_cpu(model).keys())}
    (args.output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
