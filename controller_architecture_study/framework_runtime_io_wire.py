"""Bounded command/state IPC without per-frame AirSim object pickling."""

import json
import struct
from types import SimpleNamespace

import numpy as np

from airsim_interface import RawDroneState
from .framework_runtime_wire import RuntimeConnection, frame, encode, decode


CONTEXT = ("sample_id","vehicle","model","deadline_s","status","sample_submitted_s","io_submitted_s")
RESULT = (*CONTEXT,"sample_helper_received_s","sample_io_started_s","sample_io_finished_s",
    "sample_helper_collected_s","sample_message_send_s","command_helper_collected_s","command_message_send_s",
    "command_helper_received_s","io_started_s","command_send_s","rpc_return_s","io_finished_s",
    "mapped_control","control","airsim_command")
KINDS = {"sample":b"s","send":b"c","update":b"u","sample_done":b"r","command_done":b"o"}
PACKET = struct.Struct("<QQ?d16d")


def encode_io(message):
    kind = message[0]
    if kind not in KINDS:
        return encode(message)
    row = message[-1]
    metadata = {"row":{k:row[k] for k in (CONTEXT if kind in ("sample","send") else RESULT) if k in row}}
    payload = b""
    if kind == "send" and message[1] is not None:
        action = np.asarray(message[1],dtype="<f8")
        if action.shape != (4,):
            raise ValueError("one 4D command required")
        payload = action.tobytes()
    if kind in ("sample_done","command_done") and message[1] is not None:
        raw,collision,sampled = message[1]
        values = np.r_[raw.position,raw.quaternion,raw.velocity,raw.acceleration,raw.angular_velocity]
        payload = PACKET.pack(int(raw.timestamp_ns),int(collision.time_stamp),bool(collision.has_collided),sampled,*values)
        metadata["collision_object"] = str(collision.object_name)
    return frame(KINDS[kind],metadata,payload)


def decode_io(data):
    reverse = {v:k for k,v in KINDS.items()}
    if data[:1] not in reverse:
        return decode(data)
    if len(data)<5:
        raise ValueError("truncated command IPC message")
    size = struct.unpack_from("<I",data,1)[0]
    if size>len(data)-5:
        raise ValueError("truncated command IPC metadata")
    metadata = json.loads(data[5:5+size])
    row,payload,kind = metadata["row"],data[5+size:],reverse[data[:1]]
    if kind in ("sample","update"):
        if payload:
            raise ValueError("unexpected command IPC payload")
        return (kind,row)
    if kind == "send":
        if len(payload) not in (0,32):
            raise ValueError("invalid command IPC action length")
        return (kind,np.frombuffer(payload,dtype="<f8").copy() if payload else None,row)
    packet = None
    if payload:
        if len(payload)!=PACKET.size:
            raise ValueError("invalid command IPC state length")
        stamp,collision_stamp,collided,sampled,*values = PACKET.unpack(payload)
        raw = RawDroneState(values[:3],values[3:7],values[7:10],values[10:13],values[13:16],stamp)
        collision = SimpleNamespace(time_stamp=collision_stamp,has_collided=collided,object_name=metadata["collision_object"])
        packet = raw,collision,sampled
    return (kind,packet,row)


class IOConnection(RuntimeConnection):
    encode_message = staticmethod(encode_io)
    decode_message = staticmethod(decode_io)
