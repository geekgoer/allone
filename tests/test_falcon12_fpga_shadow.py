import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from falcon12_fpga_shadow import Falcon12FpgaControlBackend, load_falcon12_fpga_config
from falcon12_fpga_udp_client import Falcon12InferenceResult


class _CpuController:
    input_dim = 12
    normalization_path = "normalization.json"
    normalization = {
        "x_mean": np.zeros(12, dtype=np.float32),
        "x_std": np.ones(12, dtype=np.float32),
    }

    def __init__(self, directory: Path, output):
        self.model_path = directory / "controller_scripted.pt"
        self.output = np.asarray(output, dtype=np.float64)
        (directory / "model_spec.json").write_text(
            json.dumps({
                "input_shape": [1, 12],
                "input_contract": "falcon_track_p_gate_relative_12d_v1",
            }),
            encoding="utf-8",
        )

    def solve_raw(self, _value):
        return self.output.copy()


class _FpgaClient:
    def __init__(self, output, *, fresh=True):
        self.output = np.asarray(output, dtype=np.float64)
        self.fresh = fresh
        self.stats = {}

    def warmup(self, _x, count, model_id):
        return [object()] * count

    def infer_normalized(self, _x, model_id, vehicle_name, cycle_index):
        del vehicle_name, cycle_index
        return Falcon12InferenceResult(
            fresh=self.fresh,
            outputs_physical=self.output if self.fresh else None,
            model_id_requested=model_id,
            model_id_received=model_id if self.fresh else None,
            sequence=7,
            request_send_timestamp_us=10,
            response_receive_timestamp_us=12 if self.fresh else None,
            rtt_us=2 if self.fresh else None,
            dpu_time_us=1 if self.fresh else None,
            server_total_us=1 if self.fresh else None,
            retries=0,
            error=None if self.fresh else "timeout",
        )


def _config(*, opt_in=False):
    return {
        "host": "192.168.77.2",
        "port": 5006,
        "protocol": "FLCN/v2",
        "model_id": 1,
        "control_period_s": 0.05,
        "udp_timeout_ms": 30.0,
        "retries": 1,
        "warmup_count": 0,
        "fallback": "cpu_dnn",
        "closed_loop_opt_in": opt_in,
    }


class Falcon12ShadowTests(unittest.TestCase):
    def test_shadow_always_applies_cpu_even_when_fpga_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = _CpuController(Path(directory), [9.8, 0.1, -0.2, 0.3])
            backend = Falcon12FpgaControlBackend(
                cpu,
                "shadow_fpga",
                _config(),
                client=_FpgaClient([15.0, 1.0, 1.0, 1.0]),
            )
            backend.warmup(np.zeros(12))
            result = backend.infer(np.zeros(12), vehicle_name="Drone1", cycle_index=0)
            np.testing.assert_allclose(result.applied_control, [9.8, 0.1, -0.2, 0.3])
            np.testing.assert_allclose(result.fpga_control, [15.0, 1.0, 1.0, 1.0])
            self.assertEqual(backend.summary()["actuator_source"], "cpu_dnn")

    def test_channel_specific_safety_limits_are_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = _CpuController(Path(directory), [25.0, 4.0, -4.0, 3.0])
            backend = Falcon12FpgaControlBackend(
                cpu,
                "shadow_fpga",
                _config(),
                client=_FpgaClient([9.8, 0.0, 0.0, 0.0]),
            )
            backend.warmup(np.zeros(12))
            result = backend.infer(np.zeros(12), vehicle_name="Drone1", cycle_index=0)
            np.testing.assert_allclose(result.applied_control, [20.0, 3.0, -3.0, 2.0])
            self.assertEqual(result.clip_flags.tolist(), [True, True, True, True])

    def test_true_fpga_mode_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = _CpuController(Path(directory), [9.8, 0.0, 0.0, 0.0])
            with self.assertRaises(ValueError):
                Falcon12FpgaControlBackend(
                    cpu,
                    "fpga",
                    _config(opt_in=False),
                    client=_FpgaClient([9.8, 0.0, 0.0, 0.0]),
                )

    def test_fpga_timeout_uses_same_cycle_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu = _CpuController(Path(directory), [9.8, 0.1, -0.2, 0.3])
            backend = Falcon12FpgaControlBackend(
                cpu,
                "fpga",
                _config(opt_in=True),
                client=_FpgaClient([0.0, 0.0, 0.0, 0.0], fresh=False),
            )
            backend.warmup(np.zeros(12))
            result = backend.infer(np.zeros(12), vehicle_name="Drone1", cycle_index=0)
            np.testing.assert_allclose(result.applied_control, [9.8, 0.1, -0.2, 0.3])
            self.assertEqual(result.fallback, "cpu_dnn")

    def test_config_loader_rejects_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = _config()
            config["protocol"] = "FLCN/v1"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_falcon12_fpga_config(path)


if __name__ == "__main__":
    unittest.main()
