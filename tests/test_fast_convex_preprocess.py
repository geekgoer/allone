"""CPU math parity checks; no AirSim, CUDA, dataset writes or training."""

from dataclasses import replace
import unittest

import numpy as np

from airsim_interface import RawDroneState
from mpc_controller import MPCReference
from falcon_paper_fidelity.motion_primitive import (ContinuousQuinticTrajectory, MotionPrimitiveConfig,
    QuinticMotionPrimitivePlanner, QuinticPolynomial, TimedQuinticSegment,
    _quaternion_from_rotation_matrix, _zup_flu_attitude)
from controller_architecture_study import fast_convex_preprocess as fast
from controller_architecture_study.convex_features import encode


def fixture():
    first = QuinticPolynomial([0., 0., -4.], [0., 0., 0.], [.2, .1, 0.],
                              [4., 1., -4.5], [3., -1., .3], [.3, -.2, 0.], 1.7)
    second = QuinticPolynomial([4., 1., -4.5], [3., -1., .3], [.3, -.2, 0.],
                               [10., -1., -4.], [0., 0., 0.], [0., 0., 0.], 2.4)
    return ContinuousQuinticTrajectory([
        TimedQuinticSegment(first, np.array([1., 0., 0., 0.]), 0.),
        TimedQuinticSegment(second, np.array([.05, 0., 0., np.sqrt(1.-.05**2)]), 1.7)])


class FastPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = fixture()
        self.planner = QuinticMotionPrimitivePlanner(MotionPrimitiveConfig())
        self.fast = fast.FastReferenceWindow(self.trajectory, self.planner.config)

    def test_rotation_conversion_all_trace_branches(self):
        rotations = np.stack([np.eye(3), np.diag([1., -1., -1.]),
                              np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])])
        expected = np.stack([_quaternion_from_rotation_matrix(r) for r in rotations])
        np.testing.assert_allclose(fast._matrix_quaternions(rotations), expected, atol=1e-14)

    def test_flatness_attitudes_including_heading_fallback(self):
        rng = np.random.default_rng(8)
        acceleration = rng.uniform(-6., 6., (100, 3))
        heading = rng.normal(size=(100, 4))
        heading[0] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
        expected = np.stack([_zup_flu_attitude(a, q, 9.8066) for a, q in zip(acceleration, heading)])
        np.testing.assert_allclose(fast._attitudes(acceleration, heading, 9.8066), expected, atol=1e-13)

    def test_reference_windows_match_at_boundaries_and_hold(self):
        phases = np.r_[0., .12345, 1.7-1e-9, 1.7, 1.7+1e-9, 4.1-1e-9, 4.1, 4.1+1.,
                       np.random.default_rng(0).uniform(0., 5., 30)]
        for elapsed in phases:
            expected = self.planner.reference_window_for_trajectory(self.trajectory,
                elapsed=float(elapsed), dt=.1, horizon=21)
            actual = self.fast.window(float(elapsed))
            self.assertEqual(len(actual), 21)
            for name in ("position", "velocity", "acceleration", "orientation", "body_rates"):
                np.testing.assert_allclose([getattr(r, name) for r in actual],
                    [getattr(r, name) for r in expected], rtol=1e-10, atol=2e-8 if name == "body_rates" else 1e-12)

    def test_direct_encoder_padding_signs_and_translation(self):
        rng = np.random.default_rng(10)
        for length in (1, 5, 21, 23):
            refs = self.fast.window(.75, horizon=length)
            refs = [replace(r, orientation=r.orientation*(-1. if i % 2 else 1.)) for i, r in enumerate(refs)]
            for _ in range(10):
                state = RawDroneState(rng.uniform(-5., 5., 3), rng.normal(size=4),
                    rng.uniform(-7., 7., 3), rng.uniform(-6., 6., 3), np.zeros(3))
                expected = encode(state, refs)
                actual = fast.encode(state, refs)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-6)
                shift = np.array([1000., -40., .146])
                moved = [replace(r, position=r.position+shift) for r in refs]
                np.testing.assert_allclose(fast.encode(replace(state, position=state.position+shift), moved),
                                           actual, rtol=0., atol=2e-6)

    def test_full_pipeline_and_teacher_outputs_match(self):
        from controller_architecture_study.convex_teacher import ConvexTeacher

        teacher = ConvexTeacher()
        for elapsed in (0., .5, 1.65, 1.7, 3.2, 4.1, 5.):
            refs = self.planner.reference_window_for_trajectory(self.trajectory,
                elapsed=elapsed, dt=.1, horizon=21)
            batched = self.fast.window(elapsed)
            state = RawDroneState(refs[0].position+[.1, .2, -.1], refs[0].orientation,
                refs[0].velocity+[.2, -.3, .1], np.zeros(3), np.zeros(3))
            np.testing.assert_allclose(fast.encode(state, batched), encode(state, refs), rtol=0., atol=2e-6)
            np.testing.assert_allclose(teacher.query(state, batched)[0], teacher.query(state, refs)[0],
                                       rtol=0., atol=2e-5)

    def test_invalid_inputs_do_not_silently_produce_features(self):
        for elapsed, dt, horizon in ((-1., .1, 21), (np.nan, .1, 21), (0., 0., 21), (0., .1, 0)):
            with self.assertRaises(ValueError):
                self.fast.window(elapsed, dt=dt, horizon=horizon)
        state = RawDroneState(np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            fast.encode(state, [MPCReference(np.zeros(3))])
        with self.assertRaises(TypeError):
            fast.encode(state, [])


if __name__ == "__main__":
    unittest.main()
