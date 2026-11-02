"""Pure NumPy scorer; no Torch/GPU dependency on the inference path."""

import json
from pathlib import Path

import numpy as np


class NumpyPredictor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.weights = {k:archive[k] for k in archive.files}

    def predict(self, state, condition):
        p = self.weights
        x = (np.asarray(state, np.float32)-p['mean'])/p['std']
        e = np.maximum(x@p['encoder.0.weight'].T+p['encoder.0.bias'], 0.)
        e = np.maximum(e@p['encoder.2.weight'].T+p['encoder.2.bias'], 0.)
        c = np.asarray(condition, np.float32)
        batch = np.concatenate((np.broadcast_to(e, (len(c), 16)), c), axis=1)
        h = np.maximum(batch@p['head.0.weight'].T+p['head.0.bias'], 0.)
        return (h@p['head.2.weight'].T+p['head.2.bias']).ravel()


class Selector:
    """Resident weights/descriptors; disk I/O happens only during startup."""
    def __init__(self, directory):
        directory = Path(directory)
        self.spec = json.loads((directory/'selector_spec.json').read_text())
        self.predictor = NumpyPredictor(directory/'weights_fp32.npz')

    def recommend(self, state, delays_s, update_s, response_s, remaining_s, selector_reserve_s):
        """Return a recommendation, never install a configuration or send control."""
        spec, names = self.spec, self.spec['models']
        if (np.asarray(state).shape != (241,) or set(delays_s) != set(names) or set(response_s) != set(names)
            or not np.isfinite(np.r_[state, list(delays_s.values()), list(response_s.values()),
                                     remaining_s, selector_reserve_s, update_s]).all()
            or any(v < 0. for v in [*delays_s.values(), *response_s.values(), selector_reserve_s])):
            raise ValueError('one finite full-state/resource context for the frozen model pool required')
        condition = np.array([spec['descriptors'][m]+[delays_s[m]/.1, update_s/.1] for m in names], np.float32)
        raw = self.predictor.predict(state, condition)
        q = spec['calibration']['correction']
        delay_min, delay_max = spec['measured_delay_range_s']
        valid = np.array([delay_min <= delays_s[m] <= delay_max for m in names])
        valid &= abs(update_s-spec['update_bound_s']) < 1e-8
        valid &= np.array([response_s[m]+selector_reserve_s <= remaining_s for m in names])
        lower = raw-(q if q is not None else np.inf)
        eligible = [m for i,m in enumerate(names) if valid[i] and lower[i] >= 0.]
        chosen = min(eligible, key=lambda m:spec['model_cost_s'][m]) if eligible else None
        return dict(model=chosen, eligible=eligible, predicted_margin=dict(zip(names, map(float, raw))),
                lower_margin={m:None if q is None else float(lower[i]) for i,m in enumerate(names)},
                calibration_available=q is not None, global_admission_still_required=True,
                physical_output_check_still_required=True, configuration_installed=False)
