"""Verify the portable M3 bundle; no Torch, AirSim, dataset or FPGA required."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_integrity(root):
    manifest = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("frozen file mismatch: " + name)
    report = json.loads((root / "deployment/export_report.json").read_text(encoding="utf-8"))
    for name, expected in report["artifact_sha256"].items():
        if digest(root / "deployment" / name) != expected:
            raise ValueError("export report hash mismatch: " + name)
    return manifest, report


def verify_golden(root, samples):
    import numpy as np
    from integer_reference import IntegerExecutor

    directory = root / "deployment"
    spec = json.loads((directory / "quantization_spec.json").read_text(encoding="utf-8"))
    for name, expected in spec["package_file_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("quantization spec hash mismatch: " + name)
    with np.load(directory / spec["weights_archive"], allow_pickle=False) as saved:
        executor = IntegerExecutor(dict(saved), spec)
    binary = (directory / spec["weights_binary"]).read_bytes()
    for name, layout in spec["array_layout"].items():
        value = np.frombuffer(binary, dtype=layout["dtype"], count=int(np.prod(layout["shape"])),
                              offset=layout["offset_bytes"]).reshape(layout["shape"])
        if value.nbytes != layout["bytes"] or not np.array_equal(value, executor.arrays[name]):
            raise ValueError("binary/archive layout mismatch: " + name)
    norm = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    with np.load(directory / "golden_vectors_int.npz", allow_pickle=False) as saved:
        if not 1 <= samples <= len(saved["X_int16"]):
            raise ValueError("samples must be between 1 and the golden row count")
        inputs = saved["X_int16"][:samples].copy()
        if not np.array_equal(executor.encode_raw(saved["X_raw"][:samples], norm), inputs):
            raise ValueError("input normalization/encoding mismatch")
        expected = saved["outputs_accumulator_int64"][:, :samples].copy()
        configurations = saved["configurations"].copy()
        exponents = saved["output_exponents"].copy()
    context = executor.context(inputs, "golden")
    context.grow(executor.max_depth, executor.max_width, sample_id="golden", model_version=executor.version)
    for i, (depth, width) in enumerate(configurations):
        depth, width = int(depth), int(width)
        codes = context.readout(depth, width)[1]
        if not np.array_equal(codes, expected[i]):
            raise ValueError("golden accumulator mismatch: d%d_w%d" % (depth, width))
        exponent = int(executor.arrays["activation_exponents"][depth]) + executor.arrays["head%d_weight_exponent" % depth]
        if not np.array_equal(exponent, exponents[i]):
            raise ValueError("golden exponent mismatch")
    for first in ((executor.max_depth, executor.widths[0]), (executor.depths[0], executor.max_width)):
        continued = executor.context(inputs, "continue")
        continued.grow(*first, sample_id="continue", model_version=executor.version)
        old = {key: value.copy() for key, value in continued.activations.items()}
        (_, codes), _ = continued.grow(executor.max_depth, executor.max_width,
                                     sample_id="continue", model_version=executor.version)
        if not np.array_equal(codes, context.readout(executor.max_depth, executor.max_width)[1]):
            raise ValueError("incremental/direct mismatch")
        if any(not np.array_equal(value, continued.activations[key]) for key, value in old.items()):
            raise ValueError("growth changed a cached prefix")
    return {"samples": samples, "configurations": len(configurations),
            "binary_archive_layout_equal": True, "integer_golden_bit_exact": True,
            "incremental_orders": 2, "cached_prefix_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", action="store_true", help="also run the NumPy integer reference")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest, report = verify_integrity(root)
    result = {"integrity": "PASS", "files": len(manifest["files"]),
              "checkpoint_sha256": report["checkpoint_sha256"],
              "model_quality_status_unchanged": report["status"], "FPGA_verified": False}
    if args.golden:
        result["software_integer_check"] = verify_golden(root, args.samples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
