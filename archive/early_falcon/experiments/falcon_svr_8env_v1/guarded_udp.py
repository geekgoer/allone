"""Reject unrepresentable current5ms INT8 inputs before sending any UDP packet."""
import numpy as np

from falcon12_fpga_udp_client import Falcon12FpgaUdpClient


class InputRangeError(RuntimeError):
    def __init__(self, detail):
        self.input_range_violation = detail
        super().__init__(f"model={detail['model_id']} INT8 input out of range at dimensions "
                         f"{detail['invalid_dimensions']}; packet NOT sent")


class GuardedFalcon12Client(Falcon12FpgaUdpClient):
    def __init__(self, *args, input_scales=None, **kwargs):
        # Confirmed by the current UDP5008 service's MODEL startup lines.
        self.input_scales = {int(k): float(v) for k, v in
                             (input_scales or {i: 32.0 for i in range(1, 7)}).items()}
        if set(self.input_scales) != set(range(1, 7)) or any(
                not np.isfinite(v) or v <= 0 for v in self.input_scales.values()):
            raise ValueError('Six finite positive input scales are required')
        super().__init__(*args, **kwargs)

    def infer_normalized(self, x_normalized, *, model_id=None, cycle_index=None):
        model_id = self.model_id if model_id is None else int(model_id)
        self._validate_model_id(model_id)
        x = np.asarray(x_normalized, dtype=np.float32).reshape(-1)
        if x.size != 12:
            raise ValueError('Expected 12 normalized input values')
        scale = self.input_scales[model_id]
        quantized = x.astype(np.float64) * scale
        invalid = np.flatnonzero(~np.isfinite(quantized) | (quantized < -128) | (quantized > 127))
        if invalid.size:
            def serial(values):
                return [float(v) if np.isfinite(v) else str(v) for v in values]
            raise InputRangeError(dict(model_id=model_id, cycle_index=cycle_index,
                scale=scale, normalized_limits=[-128/scale, 127/scale],
                invalid_dimensions=invalid.tolist(), X_normalized=serial(x),
                scaled_values=serial(quantized), packet_sent=False,
                policy='abort_trial_no_clipping_v1'))
        return super().infer_normalized(x, model_id=model_id, cycle_index=cycle_index)
