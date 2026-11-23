"""Export the release001 model-11 FP32 reference as a verified two-part chain."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ..models.plain_mlp import build_plain_mlp


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(model, sample, path, input_name, output_name):
    torch.onnx.export(model, sample, str(path), input_names=[input_name],
                      output_names=[output_name], dynamic_axes={
                          input_name: {0: "batch"}, output_name: {0: "batch"}},
                      opset_version=13, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = ROOT / "controller_architecture_study/results"
    checkpoint = (results / "decision_convex_recovery_scenic_falcon18_v1/runs"
                  / "plain_mlp_h4_w1024_seed0/best_model.pt")
    board = (ROOT / "controller_architecture_study/docs/main18_requant_release001_delivery_20261115"
             / "board_package_requant18_release001")
    normalization = board / "normalization.json"
    raw_inputs = board / "inputs_raw.txt"

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dimensions = (saved["input_dim"], saved["hidden_layers"], saved["hidden_width"])
    if dimensions != (237, 4, 1024):
        raise ValueError(f"Unexpected model dimensions {dimensions}")
    model = build_plain_mlp(torch, input_dim=237, output_dim=4, hidden_layers=4, hidden_width=1024)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    prefix, suffix = model[:4], model[4:]
    norm = json.loads(normalization.read_text(encoding="utf-8"))
    raw = np.loadtxt(raw_inputs, dtype=np.float32, ndmin=2)
    normalized = (raw - np.array(norm["x_mean"], dtype=np.float32)) / np.array(norm["x_std"], dtype=np.float32)
    y_mean, y_std = (np.array(norm[k], dtype=np.float32) for k in ("u_mean", "u_std"))
    batch = torch.from_numpy(normalized)

    args.output.mkdir(parents=True, exist_ok=False)
    whole = export(model, batch[:1], args.output / "whole.onnx", "normalized_237", "normalized_control")
    start = export(prefix, batch[:1], args.output / "prefix.onnx", "normalized_237", "activation_1024")
    with torch.no_grad():
        hidden = prefix(batch[:1])
        expected = model(batch).numpy()
    finish = export(suffix, hidden, args.output / "suffix.onnx", "activation_1024", "normalized_control")

    whole_out = whole.run(None, {"normalized_237": normalized})[0]
    activation = start.run(None, {"normalized_237": normalized})[0]
    paired = finish.run(None, {"activation_1024": activation})[0]
    # Execute two requests interleaved, then finish in reverse order with
    # independently owned activation tensors. Compare to isolated requests.
    first, second = (start.run(None, {"normalized_237": normalized[j:j+1]})[0].copy()
                     for j in (0, 1))
    for j, own in ((1, second), (0, first)):
        output = finish.run(None, {"activation_1024": own})[0]
        isolated = whole.run(None, {"normalized_237": normalized[j:j+1]})[0]
        np.testing.assert_allclose(output, isolated, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(paired, whole_out, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(paired, expected, atol=1e-5, rtol=1e-5)

    np.savez(args.output / "fp32_reference.npz", raw237=raw, normalized237=normalized,
             activation1024=activation, normalized_control=paired, physical_control=paired*y_std+y_mean)
    scaled = normalized * 16
    accepted = ((scaled > -129) & (scaled < 128) & np.isfinite(scaled)).all(axis=1)
    packed = np.pad(np.trunc(scaled[accepted]).astype(np.int8), ((0, 0), (0, 3)))
    np.savetxt(args.output / "normalized_input_int8.txt", packed, fmt="%d")
    report = {
        "scope": "same weights and FP32 CPU arithmetic, before INT8 compilation or board timing",
        "model": "main18_plain_mlp_h4_w1024_seed0",
        "cut": "after hidden layer 2 ReLU",
        "input_count": len(raw),
        "board_int8_fixture_rows": np.flatnonzero(accepted).tolist(),
        "activation_channels": int(activation.shape[1]),
        "float32_activation_bytes_per_request": int(activation[0].nbytes),
        "int8_minimum_contiguous_payload_bytes": int(activation.shape[1]),
        "maximum_abs_whole_vs_split_normalized": float(np.max(np.abs(paired-whole_out))),
        "maximum_abs_whole_vs_split_physical_by_channel": np.max(np.abs((paired-whole_out)*y_std), axis=0).tolist(),
        "maximum_abs_split_vs_pytorch_normalized": float(np.max(np.abs(paired-expected))),
        "reverse_order_interleave_passed": True,
        "inputs_sha256": {p.name: digest(p) for p in (checkpoint, normalization, raw_inputs)},
        "outputs_sha256": {p.name: digest(p) for p in sorted(args.output.iterdir())},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
