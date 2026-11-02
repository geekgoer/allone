"""Strict FLCN/v1 UDP client for the AXU3EGB FALCON DPU service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import socket
import struct
import time
from typing import Any

import numpy as np


MAGIC = 0x464C434E
VERSION = 1
INFER_REQUEST = 0x01
INFER_RESPONSE = 0x81
ERROR_RESPONSE = 0xFF
INPUT_NORMALIZED = 0x0001
HEADER_FORMAT = "!IBBHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
REQUEST_PAYLOAD_FORMAT = "!25f"
REQUEST_PAYLOAD_SIZE = struct.calcsize(REQUEST_PAYLOAD_FORMAT)
RESPONSE_PAYLOAD_FORMAT = "!I4fII"
RESPONSE_PAYLOAD_SIZE = struct.calcsize(RESPONSE_PAYLOAD_FORMAT)
ERROR_PAYLOAD_FORMAT = "!II"
UINT32_MAX = (1 << 32) - 1


class FalconUdpError(RuntimeError):
    """Base protocol/client error."""


class FalconUdpProtocolError(FalconUdpError):
    """A received datagram violated FLCN/v1."""


@dataclass(frozen=True)
class FalconInferenceResult:
    fresh: bool
    outputs_physical: np.ndarray | None
    sequence: int
    request_send_timestamp_us: int
    response_receive_timestamp_us: int | None
    rtt_us: int | None
    dpu_time_us: int | None
    server_total_us: int | None
    retries: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        output = value.pop("outputs_physical")
        value["outputs_physical"] = None if output is None else np.asarray(output).tolist()
        return value


class FalconFpgaUdpClient:
    """Persistent socket client with a global sequence space and strict replies."""

    def __init__(
        self,
        host: str = "192.168.77.2",
        port: int = 5005,
        timeout_ms: float = 10.0,
        retries: int = 1,
        *,
        sock: socket.socket | None = None,
    ) -> None:
        if not host or not (1 <= int(port) <= 65535):
            raise ValueError("host and UDP port are invalid")
        if timeout_ms <= 0.0 or retries < 0:
            raise ValueError("timeout_ms must be positive and retries non-negative")
        self.host = str(host)
        self.port = int(port)
        self.server = (self.host, self.port)
        self.timeout_seconds = float(timeout_ms) / 1000.0
        self.retries = int(retries)
        self.socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.timeout_seconds)
        self._sequence = 0
        self.stats = {
            "requests": 0,
            "fresh_responses": 0,
            "timeouts": 0,
            "retries": 0,
            "discarded": 0,
            "protocol_errors": 0,
            "server_errors": 0,
        }

    def __enter__(self) -> "FalconFpgaUdpClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.socket.close()

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & UINT32_MAX
        if self._sequence == 0:
            self._sequence = 1
        return self._sequence

    @staticmethod
    def build_request(x_normalized: Any, sequence: int, timestamp_us: int) -> bytes:
        values = np.asarray(x_normalized, dtype=np.float32).reshape(-1)
        if values.shape != (25,) or not np.isfinite(values).all():
            raise ValueError("x_normalized must contain exactly 25 finite float32 values")
        if not (1 <= int(sequence) <= UINT32_MAX) or int(timestamp_us) < 0:
            raise ValueError("sequence/timestamp are invalid")
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            INFER_REQUEST,
            INPUT_NORMALIZED,
            int(sequence),
            REQUEST_PAYLOAD_SIZE,
            int(timestamp_us),
        )
        return header + struct.pack(REQUEST_PAYLOAD_FORMAT, *values.tolist())

    def _parse_response(
        self,
        datagram: bytes,
        source: tuple[str, int],
        *,
        sequence: int,
        timestamp_us: int,
    ) -> tuple[np.ndarray, int, int]:
        if source != self.server:
            raise FalconUdpProtocolError(f"unexpected UDP source {source}")
        if len(datagram) < HEADER_SIZE:
            raise FalconUdpProtocolError(f"short header: {len(datagram)} bytes")
        magic, version, message_type, flags, reply_sequence, payload_length, reply_timestamp = struct.unpack(
            HEADER_FORMAT, datagram[:HEADER_SIZE]
        )
        if magic != MAGIC or version != VERSION or flags != 0:
            raise FalconUdpProtocolError("bad magic/version/response flags")
        if reply_sequence != sequence or reply_timestamp != timestamp_us:
            raise FalconUdpProtocolError("stale or mismatched sequence/timestamp")
        payload = datagram[HEADER_SIZE:]
        if payload_length != len(payload):
            raise FalconUdpProtocolError("payload length field does not match datagram")
        if message_type == ERROR_RESPONSE:
            if len(payload) != struct.calcsize(ERROR_PAYLOAD_FORMAT):
                raise FalconUdpProtocolError("bad error response size")
            error_code, detail = struct.unpack(ERROR_PAYLOAD_FORMAT, payload)
            self.stats["server_errors"] += 1
            raise FalconUdpProtocolError(f"server error code={error_code} detail={detail}")
        if message_type != INFER_RESPONSE or len(payload) != RESPONSE_PAYLOAD_SIZE:
            raise FalconUdpProtocolError("bad response type or payload size")
        status, thrust, wx, wy, wz, dpu_time_us, server_total_us = struct.unpack(
            RESPONSE_PAYLOAD_FORMAT, payload
        )
        if status != 0:
            raise FalconUdpProtocolError(f"nonzero inference status={status}")
        output = np.asarray((thrust, wx, wy, wz), dtype=np.float64)
        if not np.isfinite(output).all():
            raise FalconUdpProtocolError("response contains NaN or Inf")
        return output, int(dpu_time_us), int(server_total_us)

    def infer_normalized(
        self,
        x_normalized: Any,
        vehicle_name: str = "",
        cycle_index: int = 0,
    ) -> FalconInferenceResult:
        del vehicle_name, cycle_index  # metadata is logged by the caller; V1 has no vehicle field.
        sequence = self._next_sequence()
        timestamp_us = time.perf_counter_ns() // 1000
        datagram = self.build_request(x_normalized, sequence, timestamp_us)
        self.stats["requests"] += 1
        started_ns = time.perf_counter_ns()
        request_send_timestamp_us = started_ns // 1000
        last_error = "timeout"
        for attempt in range(self.retries + 1):
            if attempt:
                self.stats["retries"] += 1
            self.socket.sendto(datagram, self.server)
            deadline = time.perf_counter() + self.timeout_seconds
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self.socket.settimeout(remaining)
                try:
                    reply, source = self.socket.recvfrom(4096)
                except socket.timeout:
                    break
                try:
                    outputs, dpu_time_us, server_total_us = self._parse_response(
                        reply,
                        source,
                        sequence=sequence,
                        timestamp_us=timestamp_us,
                    )
                except FalconUdpProtocolError as exc:
                    self.stats["discarded"] += 1
                    self.stats["protocol_errors"] += 1
                    last_error = str(exc)
                    continue
                self.stats["fresh_responses"] += 1
                received_ns = time.perf_counter_ns()
                return FalconInferenceResult(
                    fresh=True,
                    outputs_physical=outputs,
                    sequence=sequence,
                    request_send_timestamp_us=request_send_timestamp_us,
                    response_receive_timestamp_us=received_ns // 1000,
                    rtt_us=(received_ns - started_ns) // 1000,
                    dpu_time_us=dpu_time_us,
                    server_total_us=server_total_us,
                    retries=attempt,
                    error=None,
                )
            self.stats["timeouts"] += 1
            last_error = "timeout waiting for matching FLCN/v1 response"
        return FalconInferenceResult(
            fresh=False,
            outputs_physical=None,
            sequence=sequence,
            request_send_timestamp_us=request_send_timestamp_us,
            response_receive_timestamp_us=None,
            rtt_us=None,
            dpu_time_us=None,
            server_total_us=None,
            retries=self.retries,
            error=last_error,
        )

    def warmup(self, x_normalized: Any, count: int = 20) -> list[FalconInferenceResult]:
        if count < 0:
            raise ValueError("warmup count must be non-negative")
        results = [self.infer_normalized(x_normalized, "warmup", -1) for _ in range(count)]
        failures = [item for item in results if not item.fresh]
        if failures:
            raise FalconUdpError(f"FPGA warm-up failed {len(failures)}/{count}: {failures[-1].error}")
        return results
