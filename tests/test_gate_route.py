import unittest

import numpy as np

from environment import generate_gate_sequence
from gate_route import ContinuousGateRoute, GatePassageError, segment_gate_crossing
from reference_generator import polyline_reference_at_time, polyline_reference_window_at_time


class GateRouteTests(unittest.TestCase):
    def test_exact_reference_is_not_shifted_by_one_control_period(self):
        reference = polyline_reference_at_time(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            elapsed=0.0,
            speed=10.0,
        )
        np.testing.assert_allclose(reference.position, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(reference.velocity, [10.0, 0.0, 0.0])

    def test_polyline_keeps_nonzero_velocity_at_intermediate_waypoint(self):
        window = polyline_reference_window_at_time(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            elapsed=0.9,
            dt=0.1,
            horizon=2,
            speed=1.0,
        )
        self.assertGreater(np.linalg.norm(window[0].velocity), 0.0)
        self.assertGreater(np.linalg.norm(window[1].velocity), 0.0)

    def test_valid_crossing_reports_offsets(self):
        gate = generate_gate_sequence(0.0, gate_count=1)[0]
        crossing = segment_gate_crossing([9.9, 0.2, -2.1], [10.1, 0.2, -2.1], gate)
        self.assertIsNotNone(crossing)
        self.assertAlmostEqual(crossing.lateral_offset, 0.2)
        self.assertAlmostEqual(crossing.vertical_offset, -0.1)

    def test_route_rejects_crossing_outside_opening(self):
        gate = generate_gate_sequence(0.0, gate_count=1)[0]
        route = ContinuousGateRoute(
            [gate],
            [8.0, 0.0, -2.0],
            approach_distance=2.0,
            exit_distance=1.0,
            reference_speed=2.0,
            start_time=0.0,
        )
        route.update([9.9, 2.0, -2.0], 0.1)
        with self.assertRaises(GatePassageError):
            route.update([10.1, 2.0, -2.0], 0.2)

    def test_route_completes_only_after_inside_plane_crossing(self):
        gate = generate_gate_sequence(0.0, gate_count=1)[0]
        route = ContinuousGateRoute(
            [gate],
            [8.0, 0.0, -2.0],
            approach_distance=2.0,
            exit_distance=1.0,
            reference_speed=2.0,
            start_time=0.0,
        )
        route.update([9.9, 0.1, -2.0], 0.1)
        crossing = route.update([10.1, 0.1, -2.0], 0.2)
        self.assertIsNotNone(crossing)
        self.assertFalse(route.completed)
        route.update([10.3, 0.1, -2.0], 0.3)
        self.assertTrue(route.completed)


if __name__ == "__main__":
    unittest.main()
