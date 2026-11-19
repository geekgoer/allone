"""Runtime wrapper for legacy and path-aware FP32 imitation controllers."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from dnn_model_utils import denormalized_output, load_normalization, normalized_input


class DNNController:
    """Load a TorchScript or PyTorch checkpoint and return physical actions."""

    def __init__(
        self,
        model_path: str | Path,
        normalization_path: str | Path | None = None,
        *,
        device: str = "cpu",
        max_thrust_accel: float = 18.0,
        max_body_rate: float = 3.0,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DNN CUDA device was requested but CUDA is unavailable")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"DNN model was not found: {self.model_path}")
        if normalization_path is None:
            normalization_path = self.model_path.parent / "normalization.json"
        self.normalization_path = Path(normalization_path).resolve()
        self.normalization = load_normalization(self.normalization_path)
        self.input_dim = int(self.normalization["x_mean"].size)
        self.reference_speed_values: tuple[float, ...] = ()
        self.input_contract: str | None = None
        model_spec_path = self.model_path.parent / "model_spec.json"
        if model_spec_path.exists():
            try:
                model_spec_payload = json.loads(model_spec_path.read_text(encoding="utf-8"))
                self.reference_speed_values = tuple(
                    float(value)
                    for value in model_spec_payload.get("reference_speed_values", [])
                )
                contract = model_spec_payload.get("input_contract")
                self.input_contract = None if contract is None else str(contract)
            except (OSError, TypeError, ValueError):
                self.reference_speed_values = ()
                self.input_contract = None
        try:
            model = torch.jit.load(str(self.model_path), map_location=self.device)
        except RuntimeError:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError(
                    "model must be TorchScript or a checkpoint containing model_state"
                )
            from dnn_model_utils import build_model

            model = build_model(
                torch,
                depth=int(checkpoint["depth"]),
                width=int(checkpoint["width"]),
                input_dim=int(checkpoint.get("input_dim", 12)),
            )
            if int(checkpoint.get("input_dim", 12)) != self.input_dim:
                raise ValueError("checkpoint and normalization use different input dimensions")
            model.load_state_dict(checkpoint["model_state"])
        self.model = model.to(self.device).eval()
        self.max_thrust_accel = float(max_thrust_accel)
        self.max_body_rate = float(max_body_rate)
        if self.max_thrust_accel <= 0.0 or self.max_body_rate <= 0.0:
            raise ValueError("DNN action limits must be positive")

    def solve_raw(self, state: Any) -> np.ndarray:
        """Return the denormalized network output before runtime safety clipping."""

        value = np.asarray(state, dtype=np.float32).reshape(-1)
        if value.size != self.input_dim or not np.isfinite(value).all():
            raise ValueError(
                f"DNN controller input must be finite with shape ({self.input_dim},)"
            )
        x_norm = normalized_input(value[None, :], self.normalization)
        with self.torch.no_grad():
            prediction_norm = self.model(self.torch.from_numpy(x_norm).to(self.device))
        prediction = denormalized_output(prediction_norm.detach().cpu().numpy(), self.normalization)[0]
        if not np.isfinite(prediction).all():
            raise RuntimeError("DNN returned NaN or Inf")
        return prediction.astype(np.float64, copy=False)

    def solve(self, state: Any) -> np.ndarray:
        prediction = self.solve_raw(state).copy()
        # Teacher actions are bounded.  Keep the same physical safety envelope
        # in case a model extrapolates slightly beyond the training support.
        prediction[0] = np.clip(prediction[0], 0.0, self.max_thrust_accel)
        prediction[1:] = np.clip(prediction[1:], -self.max_body_rate, self.max_body_rate)
        return prediction.astype(np.float64, copy=False)
