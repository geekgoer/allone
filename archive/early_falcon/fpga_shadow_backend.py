"""CPU-DNN shadow and explicit FPGA-control backend for AirSim runners."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dnn_controller import DNNController
from dnn_model_utils import normalized_input
from fpga_udp_client import FalconFpgaUdpClient, FalconInferenceResult


VALID_BACKENDS = ("cpu_dnn", "shadow_fpga", "fpga")


@dataclass(frozen=True)
class BackendCycleResult:
    applied_control: np.ndarray
    cpu_control: np.ndarray
    fpga_control: np.ndarray | None
    fpga_difference: np.ndarray | None
    fpga: FalconInferenceResult | None
    fallback: str | None
    clipped: bool
    x_normalized: np.ndarray

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "applied_control": self.applied_control.tolist(),
            "cpu_control": self.cpu_control.tolist(),
            "fpga_control": None if self.fpga_control is None else self.fpga_control.tolist(),
            "fpga_difference": None if self.fpga_difference is None else self.fpga_difference.tolist(),
            "fpga": None if self.fpga is None else self.fpga.to_dict(),
            "fallback": self.fallback,
            "clipped": self.clipped,
        }


def load_fpga_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"host", "port", "udp_timeout_ms", "retries", "warmup_count", "fallback"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"FPGA config missing keys: {missing}")
    if config["fallback"] != "cpu_dnn":
        raise ValueError("only the safe cpu_dnn fallback is supported")
    return config


class FalconFpgaControlBackend:
    """Evaluate CPU and FPGA policies while explicitly selecting the actuator source."""

    def __init__(
        self,
        cpu_controller: DNNController,
        mode: str,
        config: Mapping[str, Any],
        *,
        client: FalconFpgaUdpClient | None = None,
    ) -> None:
        if mode not in VALID_BACKENDS:
            raise ValueError(f"controller backend must be one of {VALID_BACKENDS}")
        self.cpu_controller = cpu_controller
        self.mode = mode
        self.config = dict(config)
        self.client = client
        self.owns_client = False
        if mode != "cpu_dnn" and self.client is None:
            self.client = FalconFpgaUdpClient(
                host=str(config["host"]),
                port=int(config["port"]),
                timeout_ms=float(config["udp_timeout_ms"]),
                retries=int(config["retries"]),
            )
            self.owns_client = True
        self.warmed_up = mode == "cpu_dnn"
        self.fallbacks = 0
        self.fpga_fresh = 0
        self.cycles = 0
        self.clipped_cycles = 0
        self.rtt_us: list[int] = []
        self.dpu_time_us: list[int] = []
        self.server_total_us: list[int] = []
        self.absolute_differences: list[np.ndarray] = []

    @property
    def input_dim(self) -> int:
        return self.cpu_controller.input_dim

    def close(self) -> None:
        if self.owns_client and self.client is not None:
            self.client.close()

    def normalize(self, x_physical: Any) -> np.ndarray:
        value = np.asarray(x_physical, dtype=np.float32).reshape(-1)
        if value.shape != (self.input_dim,) or not np.isfinite(value).all():
            raise ValueError(f"policy input must contain {self.input_dim} finite values")
        return normalized_input(value[None, :], self.cpu_controller.normalization)[0]

    def warmup(self, x_physical: Any) -> None:
        count = int(self.config.get("warmup_count", 20))
        x_normalized = self.normalize(x_physical)
        for _ in range(count):
            self.cpu_controller.solve(x_physical)
        if self.client is not None:
            self.client.warmup(x_normalized, count=count)
        self.warmed_up = True

    @staticmethod
    def _clip(control: Any) -> tuple[np.ndarray, bool]:
        original = np.asarray(control, dtype=np.float64).reshape(4)
        clipped = original.copy()
        clipped[0] = np.clip(clipped[0], 0.0, 18.0)
        clipped[1:] = np.clip(clipped[1:], -3.0, 3.0)
        return clipped, bool(np.any(clipped != original))

    def infer(self, x_physical: Any, *, vehicle_name: str, cycle_index: int) -> BackendCycleResult:
        if not self.warmed_up:
            raise RuntimeError("backend must be warmed up before timed inference")
        self.cycles += 1
        x_normalized = self.normalize(x_physical)
        cpu_control = self.cpu_controller.solve(x_physical)
        fpga_result = None
        fpga_control = None
        difference = None
        fallback = None
        selected = cpu_control
        if self.client is not None:
            fpga_result = self.client.infer_normalized(x_normalized, vehicle_name, cycle_index)
            if fpga_result.fresh and fpga_result.outputs_physical is not None:
                fpga_control = np.asarray(fpga_result.outputs_physical, dtype=np.float64)
                difference = fpga_control - cpu_control
                self.fpga_fresh += 1
                if fpga_result.rtt_us is not None:
                    self.rtt_us.append(fpga_result.rtt_us)
                if fpga_result.dpu_time_us is not None:
                    self.dpu_time_us.append(fpga_result.dpu_time_us)
                if fpga_result.server_total_us is not None:
                    self.server_total_us.append(fpga_result.server_total_us)
                self.absolute_differences.append(np.abs(difference))
                if self.mode == "fpga":
                    selected = fpga_control
            elif self.mode == "fpga":
                fallback = "cpu_dnn"
                self.fallbacks += 1
        applied, clipped = self._clip(selected)
        self.clipped_cycles += int(clipped)
        return BackendCycleResult(
            applied_control=applied,
            cpu_control=np.asarray(cpu_control, dtype=np.float64),
            fpga_control=fpga_control,
            fpga_difference=difference,
            fpga=fpga_result,
            fallback=fallback,
            clipped=clipped,
            x_normalized=x_normalized,
        )

    def summary(self) -> dict[str, Any]:
        def distribution(values: list[int]) -> dict[str, float | int] | None:
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

        difference = None
        if self.absolute_differences:
            values = np.vstack(self.absolute_differences)
            difference = {
                "mae_per_output": np.mean(values, axis=0).tolist(),
                "max_abs_per_output": np.max(values, axis=0).tolist(),
                "max_abs": float(np.max(values)),
            }
        return {
            "mode": self.mode,
            "cycles": self.cycles,
            "fpga_fresh": self.fpga_fresh,
            "fallbacks": self.fallbacks,
            "clipped_cycles": self.clipped_cycles,
            "rtt_us": distribution(self.rtt_us),
            "dpu_time_us": distribution(self.dpu_time_us),
            "server_total_us": distribution(self.server_total_us),
            "fpga_vs_cpu": difference,
            "client_stats": None if self.client is None else dict(self.client.stats),
        }


class BufferedJsonlLogger:
    """Keep per-cycle records out of the real-time loop's filesystem path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
