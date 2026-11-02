# Current FALCON 5 m/s 12-D six-model Ubuntu handoff

This package contains the current six models selected for redeployment. The old first model and old last model are deliberately excluded.

## Model selection

| ID | Model | Source | Best epoch | Validation normalized MSE | Test normalized MSE | Validation physical MSE | Test physical MSE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | depth10_width512 | `runs_retrain_depth10_width512_100_seed123` | 100 | 2.85133225e-04 | 2.84340989e-04 | 1.09481846e-03 | 1.08155480e-03 |
| 2 | depth10_width1024 | `runs_progressive_200_to_700_batch512_seed42/depth10_width1024` | 339 | 1.31376364e-04 | 1.32125002e-04 | 4.66094702e-04 | 4.68310609e-04 |
| 3 | depth22_width512 | `runs_progressive_200_to_700_batch512_seed42/depth22_width512` | 273 | 2.59847031e-04 | 2.64728209e-04 | 8.76730424e-04 | 8.93851684e-04 |
| 4 | depth22_width1024 | `runs_progressive_200_to_700_batch512_seed42/depth22_width1024` | 600 | 1.22835510e-04 | 1.23588252e-04 | 4.30700748e-04 | 4.32286819e-04 |
| 5 | depth34_width512 | `runs_progressive_200_to_700_batch512_seed42/depth34_width512` | 473 | 2.35607469e-04 | 2.37252607e-04 | 7.99904985e-04 | 7.97898509e-04 |
| 6 | depth34_width1024 | `runs_resume_depth34_width1024_lr5e-5_300_seed42` | 723 | 8.96891361e-05 | 8.96348283e-05 | 3.18444829e-04 | 3.17143393e-04 |

## Input and output contract

The network input is exactly 12 values in AirSim NED coordinates:

```text
X[0:3]  = p_gate - p_vehicle
X[3:6]  = shortest rotation vector of q_gate * inverse(q_vehicle)
X[6:9]  = current linear velocity
X[9:12] = current linear acceleration
```

The output order is:

```text
[collective_thrust_acceleration, body_rate_x, body_rate_y, body_rate_z]
```

Inference must use the included train-only normalization:

```text
x_norm = (x_physical - x_mean) / x_std
u_norm = DNN(x_norm)
u_physical = u_norm * u_std + u_mean
```

Do not use the old package's normalization, INT8 calibration, ELF, or golden vectors. Rebuild all deployment artifacts from this package.

## Package contents

- `models/<name>/controller_scripted.pt`: current TorchScript FP32 model.
- `models/<name>/export/weights_fp32.npz`: framework-neutral FP32 weights for rebuilding and quantization.
- `models/<name>/export/golden_vectors.npz`: FP32 parity vectors.
- `models/<name>/export/normalization.json`: model normalization.
- `dataset/train.npz`: calibration source; use only this split for INT8 calibration.
- `dataset/validation.npz` and `dataset/test.npz`: offline verification only.
- `SHA256SUMS.txt`: transfer integrity check.

## Ubuntu deployment sequence

1. Run `sha256sum -c SHA256SUMS.txt` after unpacking.
2. Rebuild each FP32 network from `export/weights_fp32.npz` in layer order. Hidden layers use ReLU; the final layer has no activation.
3. Verify each model against `export/golden_vectors.npz` before quantization.
4. Calibrate INT8 using `dataset/train.npz` only.
5. Compile the actual target DPU architecture and run board-side golden-vector verification for all six models.

The reported normalized MSE is the standardized-output MSE used by this training run. The physical MSE in the table is the directly interpretable raw-output metric. Neither should be silently presented as an exact numerical reproduction of the paper's Table II unless the paper's metric definition and data protocol are matched exactly.
