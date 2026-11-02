from pathlib import Path
import json
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1] / 'experiments/falcon_svr_8env_v1'
sys.path.insert(0, str(HERE))
import experiment as exp
from guarded_udp import GuardedFalcon12Client, InputRangeError
from falcon12_fpga_udp_client import Falcon12FpgaUdpClient


class InputGuardTests(unittest.TestCase):
    def client(self):
        return GuardedFalcon12Client(sock=MagicMock())

    def test_reproduces_minus158_without_sending(self):
        client = self.client()
        x = np.zeros(12, dtype=np.float32)
        x[1] = -4.945927619934082
        with self.assertRaises(InputRangeError) as caught:
            client.infer_normalized(x, model_id=3, cycle_index=45)
        detail = caught.exception.input_range_violation
        self.assertEqual(detail['invalid_dimensions'], [1])
        self.assertAlmostEqual(detail['scaled_values'][1], -158.26968383789062)
        self.assertEqual(detail['cycle_index'], 45)
        self.assertFalse(detail['packet_sent'])
        client.socket.sendto.assert_not_called()
        self.assertEqual(client.stats['requests'], 0)
        self.assertEqual(x[1], -4.945927619934082)

    def test_exact_boundaries_pass_unchanged(self):
        x = np.array([-4., 127/32] + [0.]*10, dtype=np.float32)
        with patch.object(Falcon12FpgaUdpClient, 'infer_normalized', return_value='ok') as infer:
            self.assertEqual(self.client().infer_normalized(x, model_id=6), 'ok')
            np.testing.assert_array_equal(infer.call_args.args[0], x)

    def test_all_dimensions_both_limits_and_nonfinite_are_guarded(self):
        client = self.client()
        for index in range(12):
            for value in (-4.01, 3.98, np.nan, np.inf, -np.inf):
                x = np.zeros(12)
                x[index] = value
                with self.assertRaises(InputRangeError) as caught:
                    client.infer_normalized(x)
                json.dumps(caught.exception.input_range_violation, allow_nan=False)
        client.socket.sendto.assert_not_called()

    def test_warmup_is_also_guarded(self):
        client = self.client()
        with self.assertRaises(InputRangeError):
            client.warmup(np.full(12, 5.), count=5, model_id=1)
        client.socket.sendto.assert_not_called()

    def test_model_specific_scale_and_invalid_scale(self):
        scales = {i: 32 for i in range(1, 7)}
        scales[2] = 64
        client = GuardedFalcon12Client(sock=MagicMock(), input_scales=scales)
        with self.assertRaises(InputRangeError):
            client.infer_normalized(np.full(12, 3.), model_id=2)
        with self.assertRaises(ValueError):
            GuardedFalcon12Client(sock=MagicMock(), input_scales={1: 0})

    def test_range_abort_is_not_collision_or_regression_target(self):
        config, _, catalog = exp.load_experiment(HERE/'collection_config_moderate_v2.json')
        job = exp.make_plan(config)[0]
        scenario = next(s for s in exp.scenarios(config) if s['id'] == job['scenario_id'])
        record = dict(job=job, scenario=scenario, wall_seconds=3, raw_log='attempt.jsonl',
            trial=dict(samples=0, completed=False, error='InputRangeError: packet NOT sent',
                       input_range_violation=dict(invalid_dimensions=[1], packet_sent=False)))
        model = next(m for m in catalog['models'] if m['model_id'] == job['model_id'])
        row = exp.outcome_row(record, model)
        self.assertEqual(row['outcome'], 'input_range_exceeded')
        self.assertFalse(row['collision'])
        self.assertIsNone(row['J_failure_zero'])
        self.assertFalse(row['valid_for_success_svr'])
        self.assertFalse(row['valid_for_failure_zero_svr'])

    def test_moderate_geometry_and_plan_keep_original_grid(self):
        config, _, _ = exp.load_experiment(HERE/'collection_config_moderate_v2.json')
        self.assertEqual(len(exp.make_plan(config)), 4608)
        self.assertEqual(len(config['environments']), 8)
        self.assertLessEqual(max(e['lateral_step_m'] for e in config['environments']), 3.2)
        self.assertGreaterEqual(min(e['x_spacing_m'] for e in config['environments']), 5.)

    def test_flight_warmup_abort_preserves_details_and_resets(self):
        import run_fpga_closed_loop as loop
        from airsim_interface import RawDroneState
        config, benchmark, _ = exp.load_experiment(HERE/'collection_config_moderate_v2.json')
        scenario = exp.scenarios(config)[0]
        client, drone, scene = MagicMock(), MagicMock(), MagicMock()
        drone.get_raw_state.return_value = RawDroneState(np.array([0., 0., -3.]),
            np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3), np.zeros(3))
        udp = self.client()
        with patch.object(loop, 'AirSimDrone', return_value=drone), \
             patch.object(loop, 'spawn_gate_sequence', return_value=scene), \
             patch.object(loop, 'vehicle_local_to_world_offset', return_value=np.zeros(3)), \
             patch.object(loop, 'cleanup_generated_gates'), patch.object(loop, '_pose'), \
             patch.object(loop.time, 'sleep'), \
             patch.object(loop, 'normalized_input', return_value=np.full((1, 12), 5.)):
            trial = loop._run_trial(client=client, airsim=MagicMock(), scenario=scenario,
                repeat_index=0, benchmark=benchmark,
                model=dict(model_id=1, model_name='depth10_width512', normalization={}),
                udp_client=udp, shadow=dict(warmup_requests_per_model=5), period_s=.06,
                actuation_delay_s=0, calibration=None, vehicle_name='Drone1',
                initial_states={scenario['initial_state_id']:object()}, output_handle=io.StringIO(),
                experiment_id='offline_test', keep_gates=False, reset_after_trial=True)
        self.assertEqual(exp.failure_kind(trial), 'input_range_exceeded')
        self.assertFalse(trial['input_range_violation']['packet_sent'])
        self.assertEqual(client.reset.call_count, 2)
        scene.destroy.assert_called_once()
        udp.socket.sendto.assert_not_called()
        json.dumps(trial, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
