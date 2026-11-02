"""Safe FLCN/v2 shadow/closed-loop backend for 12-D Track-P controllers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dnn_controller import DNNController
from dnn_model_utils import normalized_input
from falcon12_fpga_udp_client import Falcon12FpgaUdpClient, Falcon12InferenceResult


INPUT_CONTRACT = "falcon_track_p_gate_relative_12d_v1"
VALID_BACKENDS = ("shadow_fpga", "fpga")
REQUIRED_CONFIG_KEYS = {
    "host",
    "port",
    "protocol",
    "model_id",
    "control_period_s",
    "udp_timeout_ms",
    "retries",
    "warmup_count",
    "fallback",
    "closed_loop_opt_in",
}


def load_falcon12_fpga_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"FLCN/v2 config missing keys: {missing}")
    if config["protocol"] != "FLCN/v2":
        raise ValueError("12-D FPGA backend requires protocol='FLCN/v2'")
    if config["fallback"] != "cpu_dnn":
        raise ValueError("only the safe same-cycle cpu_dnn fallback is supported")
    if int(config["model_id"]) not in range(1, 7):
        raise ValueError("model_id must be in [1, 6]")
    if float(config["control_period_s"]) <= 0.0:
        raise ValueError("control_period_s must be positive")
    if float(config["udp_timeout_ms"]) <= 0.0 or int(config["retries"]) < 0:
        raise ValueError("UDP timeout/retries are invalid")
    if int(config["warmup_count"]) < 0:
        raise ValueError("warmup_count must be non-negative")
    return config


@dataclass(frozen=True)
class Falcon12BackendCycleResult:
    applied_control: np.ndarray
    cpu_control: np.ndarray
    fpga_control: np.ndarray | None
    fpga_difference: np.ndarray | None
    fpga: Falcon12InferenceResult | None
    fallback: str | None
    clipped: bool
    clip_flags: np.ndarray
    x_normalized: np.ndarray

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "applied_control": self.applied_control.tolist(),
            "cpu_control_raw": self.cpu_control.tolist(),
            "fpga_control_raw": None if self.fpga_control is None else self.fpga_control.tolist(),
            "fpga_difference_raw": (
                None if self.fpga_difference is None else self.fpga_difference.tolist()
            ),
            # Backward-compatible aliases used by existing report readers.
            "cpu_control": self.cpu_control.tolist(),
            "fpga_control": None if self.fpga_control is None else self.fpga_control.tolist(),
            "fpga_difference": (
                None if self.fpga_difference is None else self.fpga_difference.tolist()
            ),
            "fpga": None if self.fpga is None else self.fpga.to_dict(),
            "fallback": self.fallback,
            "clipped": self.clipped,
            "clip_flags": self.clip_flags.astype(bool).tolist(),
        }


class Falcon12FpgaControlBackend:
    """Compare CPU FP32 and FPGA INT8; shadow mode always applies CPU output."""

    def __init__(
        self,
        cpu_controller: DNNController,
        mode: str,
        config: Mapping[str, Any],
        *,
        client: Falcon12FpgaUdpClient | None = None,
    ) -> None:
        if mode not in VALID_BACKENDS:
            raise ValueError(f"FLCN/v2 backend mode must be one of {VALID_BACKENDS}")
        if cpu_controller.input_dim != 12:
            raise ValueError("FLCN/v2 Track-P backend requires a 12-D CPU model")
        self.cpu_controller = cpu_controller
        self.mode = mode
        self.config = dict(config)
        if self.config.get("protocol") != "FLCN/v2":
            raise ValueError("FLCN/v2 backend received a non-v2 configuration")
        self.model_id = int(self.config["model_id"])
        if self.model_id not in range(1, 7):
            raise ValueError("model_id must be in [1, 6]")
        if mode == "fpga" and not bool(self.config.get("closed_loop_opt_in", False)):
            raise ValueError(
                "FPGA closed-loop is disabled; set closed_loop_opt_in=true explicitly after shadow acceptance"
            )
        self._validate_model_contract()
        self.client = client
        self.owns_client = False
        if self.client is None:
            self.client = Falcon12FpgaUdpClient(
                host=str(self.config["host"]),
                port=int(self.config["port"]),
                timeout_ms=float(self.config["udp_timeout_ms"]),
                retries=int(self.config["retries"]),
                model_id=self.model_id,
            )
            self.owns_client = True
        self.warmed_up = False
        self.cycles = 0
        self.fpga_fresh = 0
        self.fallbacks = 0
        self.clipped_cycles = 0
        self.clip_counts = np.zeros(4, dtype=np.int64)
        self.rtt_us: list[int] = []
        self.dpu_time_us: list[int] = []
        self.server_total_us: list[int] = []
        self.absolute_differences: list[np.ndarray] = []

    @property
    def input_dim(self) -> int:
        return 12

    def _validate_model_contract(self) -> None:
        spec_path = self.cpu_controller.model_path.parent / "model_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"12-D model specification was not found: {spec_path}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("input_shape") != [1, 12]:
            raise ValueError("CPU model_spec input_shape must be [1, 12]")
        if spec.get("input_contract") != INPUT_CONTRACT:
            raise ValueError(
                f"CPU model input_contract must be {INPUT_CONTRACT!r}, got {spec.get('input_contract')!r}"
            )

    def close(self) -> None:
        if self.owns_client and self.client is not None:
            self.client.close()

    def normalize(self, x_physical: Any) -> np.ndarray:
        value = np.asarray(x_physical, dtype=np.float32).reshape(-1)
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError("Track-P policy input must contain 12 finite physical values")
        result = normalized_input(value[None, :], self.cpu_controller.normalization)[0]
        if result.shape != (12,) or not np.isfinite(result).all():
            raise ValueError("Track-P normalized input is invalid")
        return result.astype(np.float32, copy=False)

    def warmup(self, x_physical: Any) -> None:
        count = int(self.config["warmup_count"])
        x_normalized = self.normalize(x_physical)
        for _ in range(count):
            self.cpu_controller.solve_raw(x_physical)
        if self.client is not None:
            self.client.warmup(x_normalized, count=count, model_id=self.model_id)
        self.warmed_up = True

    @staticmethod
    def _clip(control: Any) -> tuple[np.ndarray, np.ndarray]:
        original = np.asarray(control, dtype=np.float64).reshape(4)
        if not np.isfinite(original).all():
            raise ValueError("control output contains NaN or Inf")
        clipped = original.copy()
        clipped[0] = np.clip(clipped[0], 2.0, 20.0)
        clipped[1] = np.clip(clipped[1], -3.0, 3.0)
        clipped[2] = np.clip(clipped[2], -3.0, 3.0)
        clipped[3] = np.clip(clipped[3], -2.0, 2.0)
        return clipped, clipped != original

    def infer(
        self,
        x_physical: Any,
        *,
        vehicle_name: str,
        cycle_index: int,
    ) -> Falcon12BackendCycleResult:
        if not self.warmed_up:
            raise RuntimeError("backend must be warmed up before timed inference")
        self.cycles += 1
        x_normalized = self.normalize(x_physical)
        cpu_control = self.cpu_controller.solve_raw(x_physical)
        fpga_result = self.client.infer_normalized(
            x_normalized,
            model_id=self.model_id,
            vehicle_name=vehicle_name,
            cycle_index=cycle_index,
        )
        fpga_control = None
        difference = None
        fallback = None
        selected = cpu_control
        if fpga_result.fresh and fpga_result.outputs_physical is not None:
            fpga_control = np.asarray(fpga_result.outputs_physical, dtype=np.float64)
            difference = fpga_control - cpu_control
            self.fpga_fresh += 1
            self.absolute_differences.append(np.abs(difference))
            if fpga_result.rtt_us is not None:
                self.rtt_us.append(fpga_result.rtt_us)
            if fpga_result.dpu_time_us is not None:
                self.dpu_time_us.append(fpga_result.dpu_time_us)
            if fpga_result.server_total_us is not None:
                self.server_total_us.append(fpga_result.server_total_us)
            if self.mode == "fpga":
                selected = fpga_control
        elif self.mode == "fpga":
            fallback = "cpu_dnn"
            self.fallbacks += 1
        # In shadow mode selected remains the same-cycle CPU FP32 output.
        applied, clip_flags = self._clip(selected)
        self.clipped_cycles += int(np.any(clip_flags))
        self.clip_counts += clip_flags.astype(np.int64)
        return Falcon12BackendCycleResult(
            applied_control=applied,
            cpu_control=np.asarray(cpu_control, dtype=np.float64),
            fpga_control=fpga_control,
            fpga_difference=difference,
            fpga=fpga_result,
            fallback=fallback,
            clipped=bool(np.any(clip_flags)),
            clip_flags=clip_flags,
            x_normalized=x_normalized,
        )

    @staticmethod
    def _distribution(values: list[int]) -> dict[str, float | int] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(np.max(array)),
        }

    def summary(self) -> dict[str, Any]:
        difference = None
        if self.absolute_differences:
            values = np.vstack(self.absolute_differences)
            difference = {
                "mae_per_output": np.mean(values, axis=0).tolist(),
                "rmse_per_output": np.sqrt(np.mean(np.square(values), axis=0)).tolist(),
                "max_abs_per_output": np.max(values, axis=0).tolist(),
                "max_abs": float(np.max(values)),
            }
        return {
            "protocol": "FLCN/v2",
            "model_id": self.model_id,
            "input_contract": INPUT_CONTRACT,
            "mode": self.mode,
            "actuator_source": "cpu_dnn" if self.mode == "shadow_fpga" else "fpga_with_cpu_fallback",
            "cycles": self.cycles,
            "fpga_fresh": self.fpga_fresh,
            "fallbacks": self.fallbacks,
            "clipped_cycles": self.clipped_cycles,
            "clip_counts_per_output": self.clip_counts.tolist(),
            "rtt_us": self._distribution(self.rtt_us),
            "dpu_time_us": self._distribution(self.dpu_time_us),
            "server_total_us": self._distribution(self.server_total_us),
            "fpga_vs_cpu_raw": difference,
            "client_stats": dict(self.client.stats),
        }
