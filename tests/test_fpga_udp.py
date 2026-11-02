import socket
import struct
import unittest

import numpy as np

from fpga_shadow_backend import FalconFpgaControlBackend
from fpga_udp_client import (
    FalconFpgaUdpClient,
    HEADER_FORMAT,
    HEADER_SIZE,
    INFER_REQUEST,
    INFER_RESPONSE,
    INPUT_NORMALIZED,
    MAGIC,
    REQUEST_PAYLOAD_SIZE,
    RESPONSE_PAYLOAD_FORMAT,
    VERSION,
)


class _RetrySocket:
    def __init__(self):
        self.sent = []
        self.timeouts = []
        self.receive_calls = 0

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendto(self, data, destination):
        self.sent.append((bytes(data), destination))

    def recvfrom(self, _size):
        self.receive_calls += 1
        if self.receive_calls == 1:
            raise socket.timeout()
        request = self.sent[-1][0]
        magic, version, _kind, _flags, sequence, _length, timestamp = struct.unpack(
            HEADER_FORMAT, request[:HEADER_SIZE]
        )
        payload = struct.pack(RESPONSE_PAYLOAD_FORMAT, 0, 9.8, 0.1, -0.2, 0.3, 1034, 1035)
        header = struct.pack(
            HEADER_FORMAT,
            magic,
            version,
            INFER_RESPONSE,
            0,
            sequence,
            len(payload),
            timestamp,
        )
        return header + payload, ("192.168.77.2", 5005)

    def close(self):
        pass


class _CpuController:
    input_dim = 25
    normalization_path = "normalization.json"
    normalization = {
        "x_mean": np.zeros(25, dtype=np.float32),
        "x_std": np.ones(25, dtype=np.float32),
    }

    @staticmethod
    def solve(_value):
        return np.asarray([9.8, 0.1, -0.2, 0.3], dtype=np.float64)


class _FpgaClient:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float64)
        self.stats = {}

    def infer_normalized(self, _x, _vehicle, _cycle):
        from fpga_udp_client import FalconInferenceResult

        return FalconInferenceResult(
            fresh=True,
            outputs_physical=self.output,
            sequence=7,
            request_send_timestamp_us=10,
            response_receive_timestamp_us=12,
            rtt_us=2,
            dpu_time_us=1,
            server_total_us=1,
            retries=0,
            error=None,
        )


class FpgaUdpTests(unittest.TestCase):
    def test_request_is_exact_network_order_contract(self):
        request = FalconFpgaUdpClient.build_request(np.arange(25), 42, 123456)
        self.assertEqual(len(request), 124)
        header = struct.unpack(HEADER_FORMAT, request[:HEADER_SIZE])
        self.assertEqual(
            header,
            (MAGIC, VERSION, INFER_REQUEST, INPUT_NORMALIZED, 42, REQUEST_PAYLOAD_SIZE, 123456),
        )
        np.testing.assert_allclose(struct.unpack("!25f", request[HEADER_SIZE:]), np.arange(25))

    def test_retry_reuses_identical_datagram_and_sequence(self):
        fake = _RetrySocket()
        client = FalconFpgaUdpClient(timeout_ms=10, retries=1, sock=fake)
        result = client.infer_normalized(np.zeros(25))
        self.assertTrue(result.fresh)
        self.assertEqual(result.sequence, 1)
        self.assertEqual(result.retries, 1)
        self.assertEqual(len(fake.sent), 2)
        self.assertEqual(fake.sent[0], fake.sent[1])
        np.testing.assert_allclose(result.outputs_physical, [9.8, 0.1, -0.2, 0.3], rtol=1e-6)

    def test_shadow_never_applies_fpga_output(self):
        backend = FalconFpgaControlBackend(
            _CpuController(),
            "shadow_fpga",
            {"warmup_count": 0},
            client=_FpgaClient([15.0, 1.0, 1.0, 1.0]),
        )
        backend.warmed_up = True
        result = backend.infer(np.zeros(25), vehicle_name="Drone1", cycle_index=0)
        np.testing.assert_allclose(result.applied_control, [9.8, 0.1, -0.2, 0.3])
        np.testing.assert_allclose(result.fpga_control, [15.0, 1.0, 1.0, 1.0])

    def test_fpga_mode_applies_fresh_output_with_limits(self):
        backend = FalconFpgaControlBackend(
            _CpuController(),
            "fpga",
            {"warmup_count": 0},
            client=_FpgaClient([19.0, 4.0, -4.0, 0.5]),
        )
        backend.warmed_up = True
        result = backend.infer(np.zeros(25), vehicle_name="Drone1", cycle_index=0)
        np.testing.assert_allclose(result.applied_control, [18.0, 3.0, -3.0, 0.5])
        self.assertTrue(result.clipped)


if __name__ == "__main__":
    unittest.main()
