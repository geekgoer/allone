"""Strict FLCN/v2 UDP client for the AXU3EGB six-model 12-D service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import socket
import struct
import time
from typing import Any

import numpy as np


MAGIC = 0x464C434E
VERSION = 2
INFER_REQUEST = 0x01
INFER_RESPONSE = 0x81
ERROR_RESPONSE = 0xFF
INPUT_NORMALIZED = 0x0001

HEADER_FORMAT = "!IBBHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
REQUEST_PAYLOAD_FORMAT = "!HH12f"
REQUEST_PAYLOAD_SIZE = struct.calcsize(REQUEST_PAYLOAD_FORMAT)
RESPONSE_PAYLOAD_FORMAT = "!HHI4fII"
RESPONSE_PAYLOAD_SIZE = struct.calcsize(RESPONSE_PAYLOAD_FORMAT)
ERROR_PAYLOAD_FORMAT = "!II"
ERROR_PAYLOAD_SIZE = struct.calcsize(ERROR_PAYLOAD_FORMAT)
REQUEST_DATAGRAM_SIZE = HEADER_SIZE + REQUEST_PAYLOAD_SIZE
RESPONSE_DATAGRAM_SIZE = HEADER_SIZE + RESPONSE_PAYLOAD_SIZE
UINT32_MAX = (1 << 32) - 1
VALID_MODEL_IDS = frozenset(range(1, 7))


class Falcon12UdpError(RuntimeError):
    """Base FLCN/v2 client error."""


class Falcon12UdpProtocolError(Falcon12UdpError):
    """A received datagram violated the FLCN/v2 contract."""


@dataclass(frozen=True)
class Falcon12InferenceResult:
    fresh: bool
    outputs_physical: np.ndarray | None
    model_id_requested: int
    model_id_received: int | None
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


def _timestamp_us() -> int:
    perf_counter_ns = getattr(time, "perf_counter_ns", None)
    if perf_counter_ns is not None:
        return int(perf_counter_ns() // 1000)
    return int(time.perf_counter() * 1_000_000)


class Falcon12FpgaUdpClient:
    """Persistent FLCN/v2 socket with strict freshness and model validation."""

    def __init__(
        self,
        host: str = "192.168.77.2",
        port: int = 5006,
        timeout_ms: float = 30.0,
        retries: int = 1,
        model_id: int = 1,
        *,
        sock: socket.socket | None = None,
    ) -> None:
        if not host or not (1 <= int(port) <= 65535):
            raise ValueError("host and UDP port are invalid")
        if timeout_ms <= 0.0 or retries < 0:
            raise ValueError("timeout_ms must be positive and retries non-negative")
        self._validate_model_id(model_id)
        self.host = str(host)
        self.port = int(port)
        self.server = (socket.gethostbyname(self.host), self.port)
        self.timeout_seconds = float(timeout_ms) / 1000.0
        self.retries = int(retries)
        self.model_id = int(model_id)
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
            "model_id_mismatches": 0,
        }

    def __enter__(self) -> "Falcon12FpgaUdpClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.socket.close()

    @staticmethod
    def _validate_model_id(model_id: int) -> None:
        if int(model_id) not in VALID_MODEL_IDS:
            raise ValueError(f"model_id must be one of {sorted(VALID_MODEL_IDS)}")

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & UINT32_MAX
        if self._sequence == 0:
            self._sequence = 1
        return self._sequence

    @staticmethod
    def build_request(
        x_normalized: Any,
        model_id: int,
        sequence: int,
        timestamp_us: int,
    ) -> bytes:
        Falcon12FpgaUdpClient._validate_model_id(model_id)
        values = np.asarray(x_normalized, dtype=np.float32).reshape(-1)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("x_normalized must contain exactly 12 finite float32 values")
        if not (1 <= int(sequence) <= UINT32_MAX) or not (0 <= int(timestamp_us) < (1 << 64)):
            raise ValueError("sequence/timestamp are invalid")
        payload = struct.pack(
            REQUEST_PAYLOAD_FORMAT,
            int(model_id),
            0,
            *values.tolist(),
        )
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            INFER_REQUEST,
            INPUT_NORMALIZED,
            int(sequence),
            len(payload),
            int(timestamp_us),
        )
        return header + payload

    def _parse_response(
        self,
        datagram: bytes,
        source: tuple[str, int],
        *,
        model_id: int,
        sequence: int,
        timestamp_us: int,
    ) -> tuple[np.ndarray, int, int, int]:
        if source != self.server:
            raise Falcon12UdpProtocolError(f"unexpected UDP source {source}")
        if len(datagram) < HEADER_SIZE:
            raise Falcon12UdpProtocolError(f"short header: {len(datagram)} bytes")
        magic, version, message_type, flags, reply_sequence, payload_length, reply_timestamp = (
            struct.unpack(HEADER_FORMAT, datagram[:HEADER_SIZE])
        )
        if magic != MAGIC:
            raise Falcon12UdpProtocolError("bad magic")
        if version != VERSION:
            raise Falcon12UdpProtocolError(f"bad protocol version {version}")
        if flags != 0:
            raise Falcon12UdpProtocolError(f"nonzero response flags {flags}")
        if reply_sequence != sequence or reply_timestamp != timestamp_us:
            raise Falcon12UdpProtocolError("stale or mismatched sequence/timestamp")
        payload = datagram[HEADER_SIZE:]
        if payload_length != len(payload):
            raise Falcon12UdpProtocolError("payload length field does not match datagram")
        if message_type == ERROR_RESPONSE:
            if len(payload) != ERROR_PAYLOAD_SIZE:
                raise Falcon12UdpProtocolError("bad error response size")
            error_code, detail = struct.unpack(ERROR_PAYLOAD_FORMAT, payload)
            self.stats["server_errors"] += 1
            raise Falcon12UdpProtocolError(f"server error code={error_code} detail={detail}")
        if message_type != INFER_RESPONSE or len(payload) != RESPONSE_PAYLOAD_SIZE:
            raise Falcon12UdpProtocolError("bad response type or payload size")
        (
            model_id_received,
            reserved,
            status,
            thrust,
            wx,
            wy,
            wz,
            dpu_time_us,
            server_total_us,
        ) = struct.unpack(RESPONSE_PAYLOAD_FORMAT, payload)
        if model_id_received != model_id:
            self.stats["model_id_mismatches"] += 1
            raise Falcon12UdpProtocolError(
                f"model_id mismatch: requested={model_id} received={model_id_received}"
            )
        if reserved != 0:
            raise Falcon12UdpProtocolError(f"nonzero response reserved field {reserved}")
        if status != 0:
            raise Falcon12UdpProtocolError(f"nonzero inference status={status}")
        output = np.asarray((thrust, wx, wy, wz), dtype=np.float64)
        if not np.isfinite(output).all():
            raise Falcon12UdpProtocolError("response contains NaN or Inf")
        return output, int(model_id_received), int(dpu_time_us), int(server_total_us)

    def infer_normalized(
        self,
        x_normalized: Any,
        model_id: int | None = None,
        vehicle_name: str = "",
        cycle_index: int = 0,
    ) -> Falcon12InferenceResult:
        del vehicle_name, cycle_index  # Logged by the caller; not part of FLCN/v2.
        selected_model_id = self.model_id if model_id is None else int(model_id)
        self._validate_model_id(selected_model_id)
        sequence = self._next_sequence()
        timestamp_us = _timestamp_us()
        datagram = self.build_request(
            x_normalized,
            selected_model_id,
            sequence,
            timestamp_us,
        )
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
                    outputs, model_id_received, dpu_time_us, server_total_us = self._parse_response(
                        reply,
                        source,
                        model_id=selected_model_id,
                        sequence=sequence,
                        timestamp_us=timestamp_us,
                    )
                except Falcon12UdpProtocolError as exc:
                    self.stats["discarded"] += 1
                    self.stats["protocol_errors"] += 1
                    last_error = str(exc)
                    continue
                self.stats["fresh_responses"] += 1
                received_ns = time.perf_counter_ns()
                return Falcon12InferenceResult(
                    fresh=True,
                    outputs_physical=outputs,
                    model_id_requested=selected_model_id,
                    model_id_received=model_id_received,
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
            last_error = "timeout waiting for matching FLCN/v2 response"
        return Falcon12InferenceResult(
            fresh=False,
            outputs_physical=None,
            model_id_requested=selected_model_id,
            model_id_received=None,
            sequence=sequence,
            request_send_timestamp_us=request_send_timestamp_us,
            response_receive_timestamp_us=None,
            rtt_us=None,
            dpu_time_us=None,
            server_total_us=None,
            retries=self.retries,
            error=last_error,
        )

    def warmup(
        self,
        x_normalized: Any,
        count: int = 20,
        model_id: int | None = None,
    ) -> list[Falcon12InferenceResult]:
        if count < 0:
            raise ValueError("warmup count must be non-negative")
        results = [self.infer_normalized(x_normalized, model_id=model_id) for _ in range(count)]
        failures = [item for item in results if not item.fresh]
        if failures:
            raise Falcon12UdpError(
                f"FPGA warm-up failed {len(failures)}/{count}: {failures[-1].error}"
            )
        return results
