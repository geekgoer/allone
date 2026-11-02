from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.framework_runtime import ModeMap, analyze_budget, route_risk, state_bin, context_grid_key, input_age_fits


class ModeContextTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"small":{"segment_estimates_s":[.001]},"large":{"segment_estimates_s":[.004]}}
        self.map = ModeMap.__new__(ModeMap)
        self.map.data = {"context_feature":"turn_pressure","analysis_to_next_command_limit_s":.080,
            "bins":{"h1_v0":{"modes":{"small":{"status":"empirical_admissible"}}}},
            "local_cells":[{"id":"against","h_range":[.25,.5],"hdot_range":[-2.5,-.8],
                "turn_pressure_range":[-3.,-.1],"modes":{"small":{"status":"empirical_admissible",
                    "delay_limit_s":.080,"future_update_limit_s":.080}}},
                {"id":"toward","h_range":[.25,.5],"hdot_range":[-2.5,-.8],
                 "turn_pressure_range":[.1,3.],"modes":{"large":{"status":"empirical_admissible",
                    "delay_limit_s":.040,"future_update_limit_s":.040}}}]}

    def test_equal_old_projection_can_have_opposite_reference_turn_pressure(self):
        state = RawDroneState(np.array([0.,.1,0.]),np.array([1.,0.,0.,0.]),
            np.array([5.,.4,0.]),np.zeros(3),np.zeros(3))
        risks = [route_risk(state,SimpleNamespace(position=np.zeros(3),velocity=np.array([5.,0.,0.]),
            acceleration=np.array([0.,a,0.])),2.) for a in (-2.,2.)]
        self.assertEqual(state_bin(risks[0]),state_bin(risks[1]))
        self.assertEqual(risks[0]["h"],risks[1]["h"])
        self.assertEqual(risks[0]["hdot"],risks[1]["hdot"])
        self.assertEqual([r["turn_pressure"] for r in risks],[-2.,2.])

    def test_context_changes_eligible_modes_and_preserves_their_distinct_limits(self):
        risk = {"h":.425,"hdot":-1.1,"turn_pressure":-2.}
        model,selection = self.map.select(risk,self.profile)
        self.assertEqual(model,"small")
        self.assertEqual(selection["delay_limit_s"],.080)
        risk["turn_pressure"] = 2.
        model,selection = self.map.select(risk,self.profile)
        self.assertEqual(model,"large")
        self.assertEqual(selection["delay_limit_s"],.040)
        self.assertEqual(selection["future_update_limit_s"],.040)

    def test_missing_unknown_context_cannot_reuse_a_positive_unconditioned_bin(self):
        for extra in ({},{"turn_pressure":float("nan")},{"turn_pressure":4.}):
            self.assertIsNone(self.map.select({"h":.425,"hdot":-1.1,**extra},self.profile)[0])

    def test_future_update_bound_is_paid_from_actuation_not_a_new_observation(self):
        state = RawDroneState(np.zeros(3),np.array([1.,0.,0.,0.]),np.zeros(3),np.zeros(3),np.zeros(3))
        _,selection = self.map.select({"h":.425,"hdot":-1.1,"turn_pressure":2.},self.profile)
        with patch("controller_architecture_study.framework_runtime.safe_horizon",return_value=(1.,[])):
            result = analyze_budget(state,[9.8,0.,0.,0.],10.020,10.,[],2.,
                maximum_update_s=selection["future_update_limit_s"])
        self.assertAlmostEqual(result["deadline_s"],10.040)

    def test_future_update_limit_must_not_silently_relax_the_global_cap(self):
        cell = self.map.data["local_cells"][1]["modes"]["large"]
        for invalid in (0.,.081,float("nan")):
            cell["future_update_limit_s"] = invalid
            with self.assertRaises(ValueError):
                self.map.select({"h":.425,"hdot":-1.1,"turn_pressure":2.},self.profile)

    def test_grid_failure_veto_cannot_fall_back_to_a_positive_local_cell(self):
        risk = {"h":.425,"hdot":-1.1,"turn_pressure":2.}
        steps = {"h":.025,"hdot":.1,"turn_pressure":.25}
        self.map.data.update(context_grid_steps=steps,context_grid={context_grid_key(risk,steps):{"modes":{}}})
        self.assertIsNone(self.map.select(risk,self.profile)[0])
        risk["h"] += .000001
        self.assertIsNone(self.map.select(risk,self.profile)[0])

    def test_input_age_is_measured_from_the_original_sample_and_reserves_command_time(self):
        row = {"sample_s":10.,"input_age_limit_s":.030}
        self.assertTrue(input_age_fits(row,10.020,.009))
        self.assertFalse(input_age_fits(row,10.020,.011))
        self.assertFalse(input_age_fits(row,10.031))
        self.assertTrue(input_age_fits({},100.))

    def test_direct_calibration_beats_a_shorter_successful_suffix_but_not_a_known_failed_wait(self):
        risk = {"h":.425,"hdot":-1.1,"turn_pressure":2.}
        steps = {"h":.025,"hdot":.1,"turn_pressure":.25}
        cell = {"modes":{"large":{"status":"empirical_admissible","delay_limit_s":.040}}}
        self.map.data.update(context_grid_steps=steps,context_grid={context_grid_key(risk,steps):cell})
        direct = self.map.data["local_cells"][1]
        direct["evidence_kind"] = "direct_recovery_calibration"
        direct["modes"]["large"]["delay_limit_s"] = .080
        self.assertEqual(self.map.select(risk,self.profile)[1]["delay_limit_s"],.080)
        cell["first_failed_wait_s"] = .060
        self.assertEqual(self.map.select(risk,self.profile)[1]["delay_limit_s"],.040)


if __name__ == "__main__":
    unittest.main()
