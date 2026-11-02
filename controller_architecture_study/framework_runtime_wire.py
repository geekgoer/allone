"""Small CPU executor messages; pickle is confined to setup/cleanup metadata."""

import json
import pickle
import struct
import time

import numpy as np


CONTEXT = ("sample_id","vehicle","model","deadline_s","priority","enqueue_s")
TIMES = ("worker_enqueue_s","first_service_s","ready_s","compute_ms","status")


def frame(kind, metadata, payload=b""):
    header = json.dumps(metadata,separators=(",",":"),allow_nan=False).encode("ascii")
    return kind+struct.pack("<I",len(header))+header+payload


def encode(message):
    kind = message[0]
    if kind == "submit":
        _,feature,row = message
        data = np.asarray(feature,dtype="<f4")
        if data.shape != (237,):
            raise ValueError("one 237D executor input required")
        return frame(b"S",{k:row[k] for k in CONTEXT},data.tobytes())
    if kind == "accepted":
        return frame(b"A",message[1:])
    if kind == "done":
        _,action,row = message
        data = b"" if action is None else np.asarray(action,dtype="<f8").tobytes()
        if data and len(data)!=32:
            raise ValueError("one 4D physical executor output required")
        return frame(b"D",{k:row[k] for k in (*CONTEXT,*TIMES) if k in row},data)
    if kind == "stop":
        return b"X"
    return b"P"+pickle.dumps(message,protocol=pickle.HIGHEST_PROTOCOL)


def decode(data):
    kind = data[:1]
    if kind == b"X" and len(data)==1:
        return ("stop",)
    if kind == b"P":
        return pickle.loads(data[1:])
    if kind not in (b"S",b"A",b"D") or len(data)<5:
        raise ValueError("invalid executor message")
    size = struct.unpack_from("<I",data,1)[0]
    if size>len(data)-5:
        raise ValueError("truncated executor metadata")
    metadata = json.loads(data[5:5+size])
    payload = data[5+size:]
    if kind == b"A":
        if payload or len(metadata)!=4:
            raise ValueError("invalid executor acknowledgement")
        return ("accepted",*metadata)
    if kind == b"S":
        if len(payload)!=237*4:
            raise ValueError("invalid executor input length")
        return ("submit",np.frombuffer(payload,dtype="<f4").copy(),metadata)
    if len(payload) not in (0,32):
        raise ValueError("invalid executor output length")
    return ("done",np.frombuffer(payload,dtype="<f8").copy() if payload else None,metadata)


class RuntimeConnection:
    encode_message = staticmethod(encode)
    decode_message = staticmethod(decode)

    def __init__(self, connection):
        self.connection = connection
        self.stalls = []

    def record(self, operation, start):
        end = time.perf_counter()
        if end-start > .005:
            self.stalls.append({"operation":operation,"start_s":start,"end_s":end})

    def send(self, message):
        start = time.perf_counter()
        data = self.encode_message(message)
        self.record("encode",start)
        start = time.perf_counter()
        self.connection.send_bytes(data)
        self.record("send",start)

    def recv(self):
        start = time.perf_counter()
        data = self.connection.recv_bytes()
        self.record("recv",start)
        start = time.perf_counter()
        message = self.decode_message(data)
        self.record("decode",start)
        return message

    def poll(self, timeout=0.):
        start = time.perf_counter()
        result = self.connection.poll(timeout)
        self.record("poll" if timeout==0. else "timed_wait",start)
        return result

    def close(self):
        self.connection.close()
