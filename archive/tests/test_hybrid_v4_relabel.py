"""Solver quality gates must reject unreliable targets rather than manufacture labels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'experiments/falcon_10ms_hybrid_160k_v1/relabel_v4_consistent.py'
SPEC = importlib.util.spec_from_file_location('hybrid_relabel', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Controller:
    def __init__(self, output=(9.8, 0, 0, 0), kkt=0):
        self.config = SimpleNamespace(min_thrust=2, max_thrust=20, max_body_rate_xy=3, max_body_rate_z=2)
        self.output = np.array(output, dtype=float)
        self.last_diagnostics = {'kkt': kkt}

    def solve(self, state, refs):
        return self.output.copy()


class QualityTests(unittest.TestCase):
    def test_requires_consecutive_stable_iterations(self):
        result = MODULE.refine(Controller(), None, None)
        self.assertTrue(result[1])
        self.assertEqual(result[2], 6)

    def test_stable_control_alone_is_not_convergence(self):
        result = MODULE.refine(Controller(kkt=12), None, None)
        self.assertFalse(result[1])
        self.assertEqual(result[2], 64)

    def test_out_of_bounds_is_not_clipped_into_validity(self):
        result = MODULE.refine(Controller(output=(30, 0, 0, 0)), None, None)
        self.assertFalse(result[1])
        self.assertEqual(result[0][0], 30)

    def test_nonfinite_kkt_is_rejected(self):
        self.assertFalse(MODULE.refine(Controller(kkt=float('nan')), None, None)[1])

    def test_solver_exception_is_a_rejected_query(self):
        c = Controller()
        def fail(*args):
            raise RuntimeError('QP failed')
        c.solve = fail
        result = MODULE.refine(c, None, None)
        self.assertFalse(result[1])
        self.assertEqual(result[5], 'QP failed')


if __name__ == '__main__':
    unittest.main()
