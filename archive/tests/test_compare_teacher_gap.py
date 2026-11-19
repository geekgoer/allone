"""Bounded diagnostics only: no model training, dataset generation, or AirSim."""

import unittest
from types import SimpleNamespace

import numpy as np

from controller_architecture_study.compare_teacher_gap import diagnostic_solve, old_formula
from controller_architecture_study.audit_capacity_20261002 import query
from mpc_controller import MPCController, MPCReference
from airsim_interface import RawDroneState


class TeacherGapTests(unittest.TestCase):
    def test_formula_matches_actual_12d_teacher_and_inactive_dimensions(self):
        controller = MPCController(max_horizontal_accel=6.)
        rng = np.random.default_rng(12)
        x = rng.uniform(-4., 4., (64, 12))
        actual = []
        for row in x:
            problem = query(controller, row)
            actual.append(controller.solve(problem.teacher_state, problem.reference_window))
        predicted, _, coefficients = old_formula(x, controller)
        np.testing.assert_allclose(predicted, actual, atol=1e-10)
        np.testing.assert_allclose(coefficients[:, [0, 9, 10, 11]], 0., atol=1e-12)
        x[:, [0, 9, 10, 11]] += 50.
        np.testing.assert_allclose(old_formula(x, controller)[0], predicted, atol=1e-12)

    def fake_teacher(self, *, alternating=False, kkt=0., output=None):
        calls = []
        controller = SimpleNamespace(last_diagnostics={})
        controller._dll = SimpleNamespace(
            falcon_acado_initialize=lambda: calls.append("initialize") or 0,
            falcon_acado_configure=lambda *args: calls.append("configure") or 0)
        controller._raise_status = lambda status, op: self.assertEqual(status, 0)
        controller.reset = lambda: calls.append("reset")

        def solve(state, references):
            count = calls.count("solve")
            calls.append("solve")
            controller.last_diagnostics = {"kkt": kkt, "objective": 1., "qp_nwsr": 0.}
            return np.asarray(output if output is not None else [9.81, 0., 0., float(count % 2) if alternating else 0.])

        controller.solve = solve
        return SimpleNamespace(controller=controller), calls

    def problem(self):
        return RawDroneState(np.zeros(3), np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3), np.zeros(3)), [MPCReference(np.zeros(3))]*21

    def test_configure_after_initialize_and_minimum_four_iterations(self):
        teacher, calls = self.fake_teacher()
        result = diagnostic_solve(teacher, *self.problem())
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 4)
        self.assertEqual(calls[:3], ["initialize", "configure", "reset"])

    def test_warm_diagnostic_reuses_solver_without_resetting(self):
        teacher, calls = self.fake_teacher()
        result = diagnostic_solve(teacher, *self.problem(), cold=False)
        self.assertTrue(result["converged"])
        self.assertEqual(calls, ["solve"]*4)

    def test_nonconvergence_and_out_of_bounds_never_accepted(self):
        for kwargs in ({"alternating": True}, {"kkt": 2.}, {"output": [21., 0., 0., 0.]}):
            teacher, _ = self.fake_teacher(**kwargs)
            result = diagnostic_solve(teacher, *self.problem(), max_iterations=8)
            self.assertFalse(result["converged"])
            self.assertEqual(result["iterations"], 8)


if __name__ == "__main__":
    unittest.main()
