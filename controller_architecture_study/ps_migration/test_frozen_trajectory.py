import copy
import json
from pathlib import Path
import unittest

from ps_core.trajectory import from_layout


class FrozenTrajectoryTests(unittest.TestCase):
    def layout(self):
        path = Path(__file__).resolve().parents[1]/'config/case.json'
        return copy.deepcopy(json.loads(path.read_text())['layouts']['Drone1'])

    def test_valid_layout_preserves_coefficients(self):
        layout = self.layout()
        window = from_layout(layout)
        self.assertEqual(window.coefficients.tolist(), [s['coefficients'] for s in layout['segments']])
        self.assertEqual(len(window.window(1.5, include_body_rates=False)), 21)

    def test_rejects_gap_bad_shape_nonfinite_and_wrong_duration(self):
        for kind in ('gap', 'shape', 'nonfinite', 'duration', 'empty'):
            with self.subTest(kind=kind):
                layout = self.layout()
                if kind == 'gap': layout['segments'][0]['start_time'] = .01
                elif kind == 'shape': layout['segments'][0]['coefficients'] = [[0.]*5]*3
                elif kind == 'nonfinite': layout['segments'][0]['coefficients'][0][0] = float('nan')
                elif kind == 'duration': layout['duration'] += .1
                else: layout['segments'] = []
                with self.assertRaises(ValueError): from_layout(layout)
