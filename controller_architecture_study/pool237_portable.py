"""NumPy-only FP32 reference for INDEPENDENT 237D controllers, not the supernet."""

import numpy as np


def predict_normalized(arrays, spec, x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 237:
        raise ValueError("expected a batch of normalized 237D inputs")
    layers = spec["linear_layers"]
    for i, layer in enumerate(layers):
        value = x @ arrays[layer["weight_key"]].T + arrays[layer["bias_key"]]
        if spec["model_type"] == "residual_mlp" and 0 < i < len(layers)-1:
            value = x + np.float32(spec["residual_scale"]) * value
        x = np.maximum(value, np.float32(0)) if i < len(layers)-1 else value
    return x


def predict_raw(arrays, spec, raw, norm):
    values = {key: np.asarray(norm[key], dtype=np.float32) for key in ("x_mean", "x_std", "u_mean", "u_std")}
    x = (np.asarray(raw, dtype=np.float32) - values["x_mean"]) / values["x_std"]
    return predict_normalized(arrays, spec, x) * values["u_std"] + values["u_mean"]
