"""Verify the six-model Ubuntu handoff package without AirSim."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = (
    ("depth10_width512", 10, 512),
    ("depth22_width512", 22, 512),
    ("depth10_width1024", 10, 1024),
    ("depth34_width512", 34, 512),
    ("depth22_width1024", 22, 1024),
    ("depth34_width1024", 34, 1024),
)


def main() -> None:
    import torch

    dataset = ROOT / "calibration"
    with np.load(dataset / "train.npz", allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        u = np.asarray(archive["U"], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 12 or u.shape != (len(x), 4):
        raise RuntimeError(f"invalid calibration arrays: X={x.shape}, U={u.shape}")

    for name, expected_depth, expected_width in MODELS:
        model_dir = ROOT / "models" / name
        spec = json.loads((model_dir / "export" / "model_spec.json").read_text())
        if spec["input_shape"] != [1, 12] or spec["output_shape"] != [1, 4]:
            raise RuntimeError(f"{name}: invalid IO shape")
        if spec["depth"] != expected_depth or spec["width"] != expected_width:
            raise RuntimeError(f"{name}: architecture mismatch")

        with np.load(model_dir / "export" / "golden_vectors.npz", allow_pickle=False) as golden:
            x_norm = np.asarray(golden["X_normalized"], dtype=np.float32)
            expected = np.asarray(golden["U_pytorch"], dtype=np.float32)
        scripted = torch.jit.load(str(model_dir / "controller_scripted.pt"), map_location="cpu").eval()
        with torch.no_grad():
            actual_norm = scripted(torch.from_numpy(x_norm)).numpy()
        normalization = json.loads((model_dir / "export" / "normalization.json").read_text())
        u_std = np.asarray(normalization["u_std"], dtype=np.float32)
        u_mean = np.asarray(normalization["u_mean"], dtype=np.float32)
        actual = actual_norm * u_std + u_mean
        max_error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"{name}: golden parity failed, max_abs_error={max_error}")
        print(f"{name}: input=12 output=4 golden_max_abs_error={max_error:.3e}")
    print(f"PASS: six models verified; calibration_samples={len(x)}")


if __name__ == "__main__":
    main()
