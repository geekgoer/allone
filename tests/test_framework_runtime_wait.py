import multiprocessing
import os
import time
import unittest

from controller_architecture_study.framework_runtime_wait import DeadlineWaiter


@unittest.skipUnless(os.name=="nt","Windows release timer")
class ReleaseWaitTests(unittest.TestCase):
    def test_message_interrupts_timer_and_timer_can_be_rearmed(self):
        receiver,sender=multiprocessing.Pipe(duplex=False)
        try:
            with DeadlineWaiter() as timer:
                sender.send_bytes(b"ready")
                ready=timer.wait([receiver],time.perf_counter()+.1)
                self.assertIn(receiver,ready)
                self.assertEqual(receiver.recv_bytes(),b"ready")
                ready=timer.wait([receiver],time.perf_counter()+.003)
                self.assertIn(timer.handle,ready)
                self.assertNotIn(receiver,ready)
            self.assertIsNone(timer.handle)
        finally:
            receiver.close()
            sender.close()

    def test_expired_release_does_not_wait(self):
        with DeadlineWaiter() as timer:
            self.assertEqual(timer.wait([],time.perf_counter()-.01),[])


if __name__=="__main__":
    unittest.main()
