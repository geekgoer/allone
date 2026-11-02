from types import SimpleNamespace
import unittest

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.framework_runtime_io_wire import encode_io, decode_io


class CommandWireTests(unittest.TestCase):
    def setUp(self):
        self.row = {"sample_id":"Drone3:7","vehicle":"Drone3","model":"frozen",
            "deadline_s":12345.67890123,"status":"ready","io_submitted_s":12345.62}

    def test_command_does_not_copy_parent_history_or_change_values(self):
        row = {**self.row,"frozen_model":"frozen","analysis_risk":{"h":.2}}
        action = np.array([9.80665,.001,-.125,2.],dtype=np.float64)
        kind,actual,result = decode_io(encode_io(("send",action,row)))
        self.assertEqual(kind,"send")
        np.testing.assert_array_equal(action,actual)
        self.assertEqual(result,self.row)
        self.assertIsNone(decode_io(encode_io(("send",None,row)))[1])

    def test_state_collision_and_nanosecond_timestamps_are_exact(self):
        raw = RawDroneState([1.,2.,3.],[1.,0.,0.,0.],[4.,5.,6.],[.1,.2,.3],[-1.,-2.,-3.],1792804079203994369)
        collision = SimpleNamespace(time_stamp=1792804079203994367,has_collided=True,object_name="Gate_横梁")
        row = {**self.row,"status":"applied","rpc_return_s":12345.625,"mapped_control":[9.8,0.,0.,0.]}
        kind,packet,result = decode_io(encode_io(("command_done",(raw,collision,12345.626),row)))
        actual,hit,sampled = packet
        self.assertEqual(kind,"command_done")
        for name in ("position","quaternion","velocity","acceleration","angular_velocity"):
            np.testing.assert_array_equal(getattr(actual,name),getattr(raw,name))
        self.assertEqual(actual.timestamp_ns,raw.timestamp_ns)
        self.assertEqual(vars(hit),vars(collision))
        self.assertEqual(sampled,12345.626)
        self.assertEqual(result,row)

    def test_ack_precedes_state_and_expired_command_has_no_packet(self):
        row = {**self.row,"status":"applied","rpc_return_s":12345.625}
        self.assertEqual(decode_io(encode_io(("update",row))),("update",row))
        expired = {**self.row,"status":"expired_not_applied"}
        self.assertEqual(decode_io(encode_io(("command_done",None,expired))),("command_done",None,expired))
        self.assertEqual(decode_io(encode_io(("sample",self.row))),("sample",self.row))

    def test_truncated_state_and_action_are_rejected(self):
        action = np.zeros(4)
        with self.assertRaises(ValueError):
            decode_io(encode_io(("send",action,self.row))[:-1])
        with self.assertRaises(ValueError):
            decode_io(b"r\xff\xff\xff\xff")

    def test_connection_uses_compact_io_codec(self):
        from controller_architecture_study.framework_runtime_io_wire import IOConnection
        class Pipe:
            def send_bytes(self,data): self.data=data
            def recv_bytes(self): return self.data
            def poll(self,timeout): return True
        pipe=Pipe()
        connection=IOConnection(pipe)
        connection.send(("send",np.zeros(4),self.row))
        self.assertEqual(pipe.data[:1],b"c")
        self.assertTrue(connection.poll())
        kind,action,row=connection.recv()
        self.assertEqual(kind,"send")
        np.testing.assert_array_equal(action,np.zeros(4))
        self.assertEqual(row,self.row)


if __name__ == "__main__":
    unittest.main()
