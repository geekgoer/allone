"""Portable integrity and FP32 golden checks; no Torch or board required."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("file mismatch: " + name)
    registry = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
    if registry["input_dim"] != 237 or registry["family"] != "independent_dense_controllers_not_supernet":
        raise ValueError("wrong model family")
    print("INTEGRITY PASS: %d models; %d files" % (len(registry["models"]), len(manifest["files"])), flush=True)
    if args.golden:
        import numpy as np
        from pool237_portable import predict_raw

        norm = json.loads((root / "shared/normalization.json").read_text(encoding="utf-8"))
        for entry in registry["models"]:
            folder = root / entry["directory"]
            spec = json.loads((folder / "model_spec.json").read_text(encoding="utf-8"))
            with np.load(folder / "weights_fp32.npz", allow_pickle=False) as arrays:
                with np.load(folder / "golden_vectors.npz", allow_pickle=False) as golden:
                    predicted = predict_raw(arrays, spec, golden["X_raw"], norm)
                    error = float(np.max(np.abs(predicted - golden["U_fp32_physical"])))
                    if not np.isfinite(error) or error > spec["numpy_golden_tolerance_physical"]:
                        raise ValueError("FP32 golden mismatch: %s %.9g" % (entry["id"], error))
            print("GOLDEN PASS %s max_abs=%.9g" % (entry["id"], error), flush=True)
    print("FP32 evidence only: DPU quantization/compilation/flight have NOT been performed by this packaging step.")


if __name__ == "__main__":
    main()
