import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from controller_architecture_study.framework_runtime_comparison_report import budget_activation, build


class ComparisonReportTests(unittest.TestCase):
    def test_old_collision_source_cannot_be_reported_as_a_verified_safety_win(self):
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root/"setup.json").write_text(json.dumps({"vehicles":7,"version":"legacy","scenarios":[{}]}))
            pair = {"scenario":{"id":"case"},"initial_state_match":True,"clean_safety_win":True,
                "static":{"collisions":1,"accepted":False},"adaptive":{"accepted":True,"total_J":1.}}
            (root/"discovery_summary.json").write_text(json.dumps({"pairs":[pair],"complete":True}))
            state = {"Drone1":{"position":[0.,0.,-2.],"velocity":[5.,0.,0.],
                "quaternion":[1.,0.,0.,0.],"held_control":[9.8,0.,0.,0.],"matched":True}}
            for policy in ("static","adaptive"):
                directory = root/"discovery/case"/policy
                directory.mkdir(parents=True)
                (directory/"initial_state.json").write_text(json.dumps(state))
            build(root,root/"report.md")
            report = json.loads((root/"report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["collision_observation_valid"])
            self.assertEqual(report["verified_clean_safety_wins"],0)

    def test_physical_budget_is_not_binding_when_a_stricter_mode_wait_limit_applies(self):
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root/"analyses.json").write_text(json.dumps([{"valid":True,"analysis_s":0.,
                "actuation_proxy_s":0.,"budget_s":.040,"mode_selection":{"delay_limit_s":.020}}]))
            result = budget_activation(root)
            self.assertEqual(result["physical_budget_binds"],0)
            self.assertEqual(result["mode_wait_limit_binds"],1)


if __name__ == "__main__":
    unittest.main()
