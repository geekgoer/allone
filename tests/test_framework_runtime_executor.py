from unittest.mock import patch
import unittest

import numpy as np

from controller_architecture_study.framework_runtime_worker import serve


class ExecutorTests(unittest.TestCase):
    def run_jobs(self,policy,deadline=2.,allow_late=False):
        class Pool:
            steps=0
            def start(self,name,feature):
                pool=self
                class Job:
                    cursor,done,output=0,False,None
                    def step(self):
                        self.cursor+=1
                        pool.steps+=1
                        self.done=self.cursor==(3 if name=="large" else 1)
                        if self.done: self.output=np.full(4,feature)
                return Job()
        pool=Pool()
        class Pipe:
            def __init__(self): self.initial=False; self.urgent=False; self.messages=[]; self.finished=0
            def poll(self,timeout=0.):
                if timeout is None:
                    raise AssertionError("idle task arrival must use the actual read, not a cancellable readiness probe")
                return not self.initial or (pool.steps>=1 and not self.urgent) or self.finished==2 or (deadline==0. and not allow_late and self.finished==1)
            def recv(self):
                if not self.initial:
                    self.initial=True
                    return ("submit",7.,{"sample_id":"low","vehicle":"a","model":"large","priority":0,"deadline_s":deadline,"enqueue_s":0.})
                if self.finished==2 or (deadline==0. and not allow_late): return ("stop",)
                self.urgent=True
                return ("submit",3.,{"sample_id":"urgent","vehicle":"b","model":"small","priority":1,"deadline_s":1.5,"enqueue_s":.1})
            def send(self,message):
                self.messages.append(message)
                if message[0]=="done": self.finished+=1
        pipe=Pipe()
        with patch("controller_architecture_study.framework_runtime_worker.time.perf_counter",return_value=1.):
            segments=serve(pipe,pool,policy,allow_late)
        return segments,pipe.messages,pool.steps

    def test_urgent_job_preempts_only_after_a_complete_segment_and_outputs_keep_context(self):
        segments,messages,_=self.run_jobs("edf")
        self.assertEqual([r["sample_id"] for r in segments],["low","urgent","low","low"])
        done={m[2]["sample_id"]:m for m in messages if m[0]=="done"}
        np.testing.assert_array_equal(done["low"][1],np.full(4,7.))
        np.testing.assert_array_equal(done["urgent"][1],np.full(4,3.))
        self.assertEqual(done["low"][2]["deadline_s"],2.)

    def test_static_priority_uses_the_same_segment_boundaries(self):
        segments,_,_=self.run_jobs("static")
        self.assertEqual([r["sample_id"] for r in segments],["low","low","low","urgent"])

    def test_expired_job_executes_no_segment_and_returns_no_action(self):
        segments,messages,steps=self.run_jobs("edf",0.)
        self.assertFalse(segments)
        self.assertEqual(steps,0)
        done=[m for m in messages if m[0]=="done"]
        self.assertIsNone(done[0][1])
        self.assertEqual(done[0][2]["status"],"expired_before_segment")

    def test_exploration_finishes_expired_job_without_extending_deadline(self):
        segments,messages,steps=self.run_jobs("edf",0.,allow_late=True)
        self.assertEqual(steps,4)
        done={m[2]["sample_id"]:m for m in messages if m[0]=="done"}
        np.testing.assert_array_equal(done["low"][1],np.full(4,7.))
        self.assertEqual(done["low"][2]["deadline_s"],0.)
        self.assertEqual(len(segments),4)


if __name__=="__main__":
    unittest.main()
