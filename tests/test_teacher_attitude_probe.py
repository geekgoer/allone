"""Physics/control semantics for the isolated teacher comparison, without hardware."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from controller_architecture_study.convex_teacher import ConvexTeacher
from controller_architecture_study.teacher_attitude_probe import (
    DiagnosticTeacher, VARIANTS, desired_attitude, qp_feedforward)
from state_adapter import quaternion_multiply


def problem():
    state = RawDroneState(np.array([0., .3, 0.]), np.array([1., 0., 0., 0.]),
                          np.zeros(3), np.zeros(3), np.zeros(3))
    refs = [MPCReference(np.array([i*.5, 0., 0.]), np.array([1.,0.,0.,0.]),
                         np.array([5.,0.,0.]), np.zeros(3), np.zeros(3)) for i in range(21)]
    return state, refs


class AttitudeDiagnosticTests(unittest.TestCase):
    def test_baseline_preserves_frozen_teacher_outputs_exactly(self):
        old, new = ConvexTeacher(), DiagnosticTeacher('baseline')
        state, refs = problem()
        for delta in (np.zeros(3), np.array([.2, -.1, .1]), np.array([-.4, .3, -.1])):
            query = replace(state, position=state.position+delta)
            a, _ = old.query(query, refs)
            b, _ = new.query(query, refs)
            np.testing.assert_array_equal(a, b)

    def test_gain_change_does_not_change_qp_or_thrust(self):
        state, refs = problem()
        a, ai = DiagnosticTeacher('baseline').query(state, refs)
        b, bi = DiagnosticTeacher('gain7').query(state, refs)
        self.assertFalse(ai['rate_clipped'])
        np.testing.assert_array_equal(ai['first_acceleration'], bi['first_acceleration'])
        self.assertEqual(a[0], b[0])
        np.testing.assert_allclose(b[1:], np.clip(2*a[1:], [-3.,-3.,-2.], [3.,3.,2.]), atol=1e-6)

    def test_feedforward_is_transported_from_desired_to_current_body(self):
        q0 = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)])
        q1 = quaternion_multiply(q0, [np.cos(.05), np.sin(.05), 0., 0.])
        ff = qp_feedforward(np.array([1.,0.,0.,0.]), q0, q1, .1)
        np.testing.assert_allclose(ff, [0., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(qp_feedforward(-np.array([1.,0.,0.,0.]), -q0, q1, .1), ff, atol=1e-12)

    def test_gain5_changes_only_the_feedback_gain(self):
        state, refs = problem()
        old, oi = DiagnosticTeacher('baseline').query(state, refs)
        teacher = DiagnosticTeacher('gain5')
        action, info = teacher.query(state, refs)
        self.assertEqual(teacher.settings.attitude_gain, 5.)
        self.assertFalse(teacher.use_feedforward)
        np.testing.assert_array_equal(oi['first_acceleration'], info['first_acceleration'])
        self.assertEqual(old[0], action[0])
        np.testing.assert_allclose(action[1:], np.clip(5*info['attitude_error_rad'],
            [-3.,-3.,-2.], [3.,3.,2.]), atol=1e-6)
        self.assertEqual(teacher.signature()['diagnostic_variant'], 'gain5')

    def test_explicit_diagnostic_gain_is_recorded_and_preserves_qp(self):
        state, refs = problem()
        old, oi = DiagnosticTeacher('baseline').query(state, refs)
        teacher = DiagnosticTeacher('baseline', attitude_gain=4.5)
        action, info = teacher.query(state, refs)
        self.assertEqual(teacher.signature()['settings']['attitude_gain'], 4.5)
        np.testing.assert_array_equal(oi['first_acceleration'], info['first_acceleration'])
        self.assertEqual(old[0], action[0])
        np.testing.assert_allclose(action[1:], np.clip(4.5*info['attitude_error_rad'],
            [-3.,-3.,-2.], [3.,3.,2.]), atol=1e-6)
        for invalid in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                DiagnosticTeacher('baseline', attitude_gain=invalid)

    def test_unchanging_desired_attitude_has_zero_feedforward(self):
        q = np.array([1.,0.,0.,0.])
        np.testing.assert_array_equal(qp_feedforward(q, q, q, .1), np.zeros(3))

    def test_all_variants_preserve_hover(self):
        state, refs = problem()
        state = replace(state, position=np.zeros(3))
        refs = [replace(r, position=np.zeros(3), velocity=np.zeros(3)) for r in refs]
        for variant in VARIANTS:
            action, info = DiagnosticTeacher(variant).query(state, refs)
            np.testing.assert_allclose(action, [9.80665,0.,0.,0.], atol=1e-5)
            np.testing.assert_allclose(info['feedforward_rates'], np.zeros(3), atol=1e-10)

    def test_all_variants_use_one_qp_and_the_same_optimized_acceleration(self):
        state, refs = problem()
        first = None
        for variant in VARIANTS:
            teacher = DiagnosticTeacher(variant)
            with patch.object(teacher.solver.inner, 'solve', wraps=teacher.solver.inner.solve) as solve:
                action, info = teacher.query(state, refs)
            solve.assert_called_once()
            if first is None:
                first = info['first_acceleration']
            np.testing.assert_array_equal(info['first_acceleration'], first)
            expected = desired_attitude(info['next_acceleration'], refs[1].orientation, teacher.settings.gravity)
            np.testing.assert_allclose(info['next_desired_quaternion'], expected, atol=1e-12)
            if teacher.use_feedforward:
                np.testing.assert_allclose(action[1:], np.clip(info['feedback_rates']+info['feedforward_rates'],
                    [-3.,-3.,-2.], [3.,3.,2.]), atol=1e-6)
            json.dumps({k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()})

    def test_planner_body_rate_is_not_mixed_into_qp_feedforward(self):
        teacher = DiagnosticTeacher('qp_ff')
        state, refs = problem()
        a, _ = teacher.query(state, refs)
        b, _ = teacher.query(state, [replace(r, body_rates=np.array([3.,-3.,2.])) for r in refs])
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
