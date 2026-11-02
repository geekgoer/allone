import unittest
from unittest.mock import patch

from controller_architecture_study.framework_cpu_affinity import choose_layout, CpuPlacement


def topology():
    return [{"group":0,"logical":i,"core":i//2,"efficiency_class":1} for i in range(16)]+[
        {"group":0,"logical":i,"core":i,"efficiency_class":0} for i in range(16,32)]


class AffinityTests(unittest.TestCase):
    def test_distinct_physical_cores_and_smt_exclusion(self):
        rows = topology()
        result = choose_layout(rows,list(range(32)))
        chosen = result["inference_cpus"]+result["frontend_cpus"]+result["io_cpus"]
        cores = {r["logical"]:r["core"] for r in rows}
        self.assertEqual(len({cores[c] for c in chosen}),4)
        self.assertEqual(len(result["reserved_physical_siblings"]),8)
        self.assertFalse(set(result["inference_cpus"]) & set(result["parent_process_cpus"]))
        self.assertTrue(all(c<16 for c in chosen))

    def test_rejects_insufficient_physical_cores(self):
        with self.assertRaises(ValueError):
            choose_layout(topology(),list(range(8)))

    def test_rejects_multiple_processor_groups(self):
        with self.assertRaises(ValueError):
            choose_layout([{"group":1}],list(range(32)))

    def test_no_affinity_change_when_disabled(self):
        with patch("controller_architecture_study.framework_cpu_affinity.set_process_cpus") as set_mask:
            with CpuPlacement(False) as scope:
                self.assertFalse(scope.enabled)
            set_mask.assert_not_called()
            self.assertTrue(scope.report["restoration_passed"])

    def test_parent_restored_even_when_trial_raises(self):
        with patch("controller_architecture_study.framework_cpu_affinity.set_process_cpus") as process, patch(
                "controller_architecture_study.framework_cpu_affinity.set_thread_cpus") as thread:
            scope = CpuPlacement()
            scope.parent = {"previous":[0,1,2,3]}
            scope.thread = {"previous":[0,1]}
            self.assertFalse(scope.__exit__(RuntimeError,RuntimeError("trial"),None))
            process.assert_called_once_with([0,1,2,3])
            thread.assert_called_once_with([0,1])


if __name__ == "__main__":
    unittest.main()
