"""Offline checks of the natural-route experiment; no hardware or flying."""
import unittest

import numpy as np

from controller_architecture_study.ip37_safety_cases import audit, evaluate, make_case


class SafetyCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.case = make_case()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(f'Frozen validation routes must be supplied: {exc}')

    def test_reference_is_continuous_and_clear_of_inflated_frames(self):
        report = audit(self.case)
        self.assertTrue(report['passed'])
        for result in report['vehicles'].values():
            self.assertLess(result['max_C2_error'], 1e-6)
            self.assertLessEqual(result['max_sampled_speed_mps'], 8.)
            self.assertLessEqual(result['max_sampled_acceleration_mps2'], 8.)

    def test_turn_gates_require_acceleration_and_face_horizontal_velocity(self):
        for name in ('Drone1', 'Drone2', 'Drone3'):
            gates = [g for g in self.case['layouts'][name]['gates'] if g['kind'] == 'turn']
            self.assertEqual(len(gates), 2)
            for g in gates:
                v, a = np.array(g['reference_velocity']), np.array(g['reference_acceleration'])
                self.assertGreater(np.linalg.norm(np.cross(v, a)), 1.)
                w, _, _, z = g['orientation']
                normal = np.array([w*w-z*z, 2*w*z])
                self.assertAlmostEqual(float(normal @ (v[:2]/np.linalg.norm(v[:2]))), 1.)

    def test_dangerous_gate_times_are_staggered_without_different_starts(self):
        times = [sorted(self.case['roles'][f'Drone{i}']['turn_gate_times_s']) for i in (1, 2, 3)]
        np.testing.assert_allclose(np.array(times[1])-times[0], 6.)
        np.testing.assert_allclose(np.array(times[2])-times[0], 12.)
        for layout in self.case['layouts'].values():
            np.testing.assert_allclose(evaluate(layout['segments'][0], 0.), [0., 0., -4.])
            np.testing.assert_allclose(evaluate(layout['segments'][0], 0., 1), np.zeros(3))
        self.assertEqual(self.case['duration_s'], 26.)

    def test_seed_rotates_roles_without_changing_route_difficulty(self):
        other = make_case(seed=1)
        for name in ('Drone1', 'Drone2', 'Drone3'):
            self.assertNotEqual(self.case['roles'][name]['role'], other['roles'][name]['role'])
        self.assertEqual(sorted(r['role'] for r in other['roles'].values()),
                         sorted(r['role'] for r in self.case['roles'].values()))
        self.assertFalse(other['external_disturbance'])
        self.assertFalse(other['live_flight_verified'])


if __name__ == '__main__':
    unittest.main()
