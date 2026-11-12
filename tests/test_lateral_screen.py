import json
from pathlib import Path
import unittest
import numpy as np
from controller_architecture_study.screen_lateral_cases import transform
from controller_architecture_study.ip37_safety_cases import evaluate


class LateralScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path('controller_architecture_study/results/ip37_safety_design_20261107/seed0_w21/case.json')
        cls.source = json.loads(path.read_text(encoding='utf-8'))

    def test_reference_timing_and_derivatives_remain_consistent(self):
        case = transform(self.source, .7)
        old, new = self.source['layouts']['Drone1'], case['layouts']['Drone1']
        self.assertEqual(old['duration'], new['duration'])
        self.assertEqual([g['reference_time_s'] for g in old['gates']],
                         [g['reference_time_s'] for g in new['gates']])
        origin = old['segments'][0]['coefficients'][1][0]
        for a, b in zip(old['segments'], new['segments']):
            self.assertEqual(a['start_time'], b['start_time'])
            for derivative in (0, 1, 2):
                x, y = evaluate(a, a['duration']/2, derivative), evaluate(b, b['duration']/2, derivative)
                expected = x.copy()
                expected[1] = origin+.7*(x[1]-origin) if derivative == 0 else .7*x[1]
                np.testing.assert_allclose(y, expected, atol=1e-9)
        self.assertEqual(case['net_width_m'], 3.)
        self.assertEqual(self.source['net_width_m'], 2.1)

    def test_scale_one_preserves_original_trajectory_and_invalid_scale_fails(self):
        case = transform(self.source, 1.)
        self.assertEqual(case['layouts'], self.source['layouts'])
        for x in (0., -1., 1.1, float('nan')):
            with self.assertRaises(ValueError):
                transform(self.source, x)
