from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import fly_ip37_policy as pilot


class BoardFlightTests(unittest.TestCase):
    def test_board_output_is_forwarded_to_existing_flight_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / 'inputs/11.txt'
            golden.parent.mkdir()
            action = np.array([9.8, .1, -.15, .02])
            np.savetxt(golden, np.r_[np.zeros(237), action, action][None, :])
            board = SimpleNamespace(close=lambda: None, peer=('192.168.77.2', 5238))
            args = SimpleNamespace(period_ms=50., output=root/'out', bundle=root, board='192.168.77.2',
                board_port=5238, model_id=11, environment_id=48, route_index=0,
                vehicle='Drone1', airsim_host='127.0.0.1')
            first = SimpleNamespace(position=np.array([0., 0., -4.]),
                                    orientation=np.array([1., 0., 0., 0.]))
            planner = SimpleNamespace(trajectory=SimpleNamespace(duration=12.),
                reference_window_for_trajectory=lambda *a, **k: [first])
            rpc = SimpleNamespace(confirmConnection=lambda: None, listVehicles=lambda: ['Drone1'])
            api = SimpleNamespace(MultirotorClient=lambda **k: rpc)
            def flight(client, pool, layout, start, trial, options, output):
                np.testing.assert_allclose(pool.predict(trial['model'], np.zeros(237))[0], action)
                self.assertEqual(start['velocity'], [0., 0., 0.])
                self.assertEqual(options.vehicle, 'Drone1')
                return dict(error=None, comparable=True, window_covered=True, skipped_releases=0,
                            crossings=[dict(valid=True) for _ in range(5)])
            with patch.object(pilot, 'make_client', return_value=(board, {
                'models': [{'numeric_id': 11}], 'wire_tag_hex': 'candidate'})), \
                 patch.object(pilot, 'predict', return_value=(action, {})) as predict, \
                 patch.object(pilot, 'load_layout', return_value=({}, 'route')), \
                 patch.object(pilot, 'StoredRoutePlanner', return_value=planner), \
                 patch.object(pilot, 'load_airsim_client'), \
                 patch('airsim_interface.require_airsim', return_value=api), \
                 patch.object(pilot, 'flight', side_effect=flight):
                result = pilot.run(args)
            self.assertTrue(result['flight_passed'])
            self.assertTrue(result['outputs_applied'])
            self.assertEqual(result['mpc_fallbacks'], 0)
            self.assertEqual(predict.call_count, 2)


if __name__ == '__main__':
    unittest.main()
