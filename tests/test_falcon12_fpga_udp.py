import socket
import struct
import unittest

import numpy as np

from falcon12_fpga_udp_client import (
    ERROR_RESPONSE,
    Falcon12FpgaUdpClient,
    HEADER_FORMAT,
    HEADER_SIZE,
    INFER_REQUEST,
    INFER_RESPONSE,
    INPUT_NORMALIZED,
    MAGIC,
    REQUEST_DATAGRAM_SIZE,
    REQUEST_PAYLOAD_FORMAT,
    REQUEST_PAYLOAD_SIZE,
    RESPONSE_DATAGRAM_SIZE,
    RESPONSE_PAYLOAD_FORMAT,
    VERSION,
)


class _ScriptedSocket:
    def __init__(self, reply_builders):
        self.reply_builders = list(reply_builders)
        self.sent = []
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendto(self, data, destination):
        self.sent.append((bytes(data), destination))

    def recvfrom(self, _size):
        if not self.reply_builders:
            raise socket.timeout()
        builder = self.reply_builders.pop(0)
        if builder is None:
            raise socket.timeout()
        return builder(self.sent[-1][0])

    def close(self):
        pass


def _success_reply(request, *, model_id=1, source=("192.168.77.2", 5006)):
    magic, version, _kind, _flags, sequence, _length, timestamp = struct.unpack(
        HEADER_FORMAT, request[:HEADER_SIZE]
    )
    payload = struct.pack(
        RESPONSE_PAYLOAD_FORMAT,
        model_id,
        0,
        0,
        10.25,
        -0.5,
        1.25,
        0.125,
        1035,
        1038,
    )
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
    return header + payload, source


class Falcon12FpgaUdpTests(unittest.TestCase):
    def test_request_matches_76_byte_network_order_contract(self):
        values = np.arange(12, dtype=np.float32) / 10.0
        request = Falcon12FpgaUdpClient.build_request(values, 6, 42, 123456)
        self.assertEqual(len(request), REQUEST_DATAGRAM_SIZE)
        self.assertEqual(REQUEST_DATAGRAM_SIZE, 76)
        header = struct.unpack(HEADER_FORMAT, request[:HEADER_SIZE])
        self.assertEqual(
            header,
            (MAGIC, VERSION, INFER_REQUEST, INPUT_NORMALIZED, 42, REQUEST_PAYLOAD_SIZE, 123456),
        )
        model_id, reserved, *payload_values = struct.unpack(
            REQUEST_PAYLOAD_FORMAT, request[HEADER_SIZE:]
        )
        self.assertEqual((model_id, reserved), (6, 0))
        np.testing.assert_allclose(payload_values, values)

    def test_success_response_matches_56_byte_contract(self):
        fake = _ScriptedSocket([_success_reply])
        client = Falcon12FpgaUdpClient(timeout_ms=10, retries=0, model_id=1, sock=fake)
        result = client.infer_normalized(np.zeros(12))
        self.assertEqual(len(_success_reply(fake.sent[0][0])[0]), RESPONSE_DATAGRAM_SIZE)
        self.assertEqual(RESPONSE_DATAGRAM_SIZE, 56)
        self.assertTrue(result.fresh)
        self.assertEqual(result.model_id_requested, 1)
        self.assertEqual(result.model_id_received, 1)
        np.testing.assert_allclose(result.outputs_physical, [10.25, -0.5, 1.25, 0.125])

    def test_retry_reuses_identical_datagram_sequence_and_timestamp(self):
        fake = _ScriptedSocket([None, _success_reply])
        client = Falcon12FpgaUdpClient(timeout_ms=10, retries=1, model_id=1, sock=fake)
        result = client.infer_normalized(np.zeros(12))
        self.assertTrue(result.fresh)
        self.assertEqual(result.sequence, 1)
        self.assertEqual(result.retries, 1)
        self.assertEqual(len(fake.sent), 2)
        self.assertEqual(fake.sent[0], fake.sent[1])

    def test_wrong_model_reply_is_discarded_before_matching_reply(self):
        def wrong_model(request):
            return _success_reply(request, model_id=2)

        fake = _ScriptedSocket([wrong_model, _success_reply])
        client = Falcon12FpgaUdpClient(timeout_ms=10, retries=0, model_id=1, sock=fake)
        result = client.infer_normalized(np.zeros(12))
        self.assertTrue(result.fresh)
        self.assertEqual(client.stats["discarded"], 1)
        self.assertEqual(client.stats["model_id_mismatches"], 1)

    def test_wrong_source_is_discarded(self):
        def wrong_source(request):
            return _success_reply(request, source=("192.168.77.3", 5006))

        fake = _ScriptedSocket([wrong_source, _success_reply])
        client = Falcon12FpgaUdpClient(timeout_ms=10, retries=0, model_id=1, sock=fake)
        result = client.infer_normalized(np.zeros(12))
        self.assertTrue(result.fresh)
        self.assertEqual(client.stats["discarded"], 1)

    def test_error_response_is_never_accepted_as_control(self):
        def error_reply(request):
            _magic, _version, _kind, _flags, sequence, _length, timestamp = struct.unpack(
                HEADER_FORMAT, request[:HEADER_SIZE]
            )
            payload = struct.pack("!II", 7, 99)
            header = struct.pack(
                HEADER_FORMAT,
                MAGIC,
                VERSION,
                ERROR_RESPONSE,
                0,
                sequence,
                len(payload),
                timestamp,
            )
            return header + payload, ("192.168.77.2", 5006)

        fake = _ScriptedSocket([error_reply, None])
        client = Falcon12FpgaUdpClient(timeout_ms=10, retries=0, model_id=1, sock=fake)
        result = client.infer_normalized(np.zeros(12))
        self.assertFalse(result.fresh)
        self.assertIsNone(result.outputs_physical)
        self.assertEqual(client.stats["server_errors"], 1)

    def test_invalid_input_and_model_id_are_rejected(self):
        with self.assertRaises(ValueError):
            Falcon12FpgaUdpClient.build_request(np.zeros(25), 1, 1, 1)
        with self.assertRaises(ValueError):
            Falcon12FpgaUdpClient.build_request(np.zeros(12), 0, 1, 1)
        with self.assertRaises(ValueError):
            Falcon12FpgaUdpClient.build_request(np.full(12, np.nan), 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
