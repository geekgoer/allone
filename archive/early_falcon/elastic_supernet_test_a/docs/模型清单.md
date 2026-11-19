# Test A model inventory

## Selected immutable baseline

- Checkpoint: `experiments/paper_track_p_12d_5ms_v1/runs_serial_v2/paper_track_p_12d_5ms_v1_serial_v2_depth22_width512/best_model.pt`
- Input: 12 physical values in AirSim NED coordinates.
- Input contract: `[p_gate-p, r_gate-r, velocity, acceleration]`.
- Normalization: external affine input normalization and external affine output de-normalization.
- Graph: 22 fully connected layers in total.
- Hidden layers: 21 layers, each `Linear(..., 512) + ReLU`.
- Output layer: `Linear(512, 4)`, without an output activation.
- Output: collective thrust acceleration and three body rates.
- Dropout: none.
- Residual connections: none.
- Other non-FC operations: only ReLU and external normalization.
- Stored format: PyTorch checkpoint dictionary containing a Sequential-model `model_state_dict` and training metadata.
- Parameters: 5,261,828.
- Existing offline test result: physical MSE `8.2552717e-05`, RMSE `0.00908585` over 48,000 fixed test samples.

## Weight shapes

- Linear 0: `[512, 12]`, bias `[512]`.
- Linear 1 through 20: each `[512, 512]`, bias `[512]`.
- Linear 21: `[4, 512]`, bias `[4]`.

## Initial elastic search space

The implementation accepts one active width per hidden layer. The first enumerable experiment uses a single uniform active width across all 21 hidden layers:

`16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512`.

All widths are multiples of 16. Each subnet uses prefix slices of one maximum parameter set. This global-width sweep is the first controlled experiment; stage-wise and per-layer search can be added without changing the elastic operator.

