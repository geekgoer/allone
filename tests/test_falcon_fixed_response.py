import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_falcon_paper_metric import CLOSED_LOOP, MODULE_DIR

spec = importlib.util.spec_from_file_location('capacity_axis_check', MODULE_DIR / 'run_capacity_axis_check.py')
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)


class FixedResponseTests(unittest.TestCase):
    def test_fast_and_slow_inference_wait_to_same_send_time(self):
        for native_us in (6_000, 18_000):
            clock = [1_000_000 + native_us]
            def sleep(seconds):
                clock[0] += round(seconds * 1_000_000)
            with patch.object(CLOSED_LOOP.time, 'perf_counter_ns', side_effect=lambda: clock[0] * 1000), patch.object(CLOSED_LOOP.time, 'sleep', side_effect=sleep):
                CLOSED_LOOP._wait_until_us(1_025_000)
            self.assertEqual(clock[0], 1_025_000)

    def test_late_inference_is_not_delayed_further(self):
        with patch.object(CLOSED_LOOP.time, 'perf_counter_ns', return_value=1_030_000_000), patch.object(CLOSED_LOOP.time, 'sleep') as sleep:
            CLOSED_LOOP._wait_until_us(1_025_000)
            sleep.assert_not_called()
        self.assertTrue(CLOSED_LOOP._response_target_violation(30_000, 25, 3))

    def test_recorded_latency_band_includes_send_call(self):
        self.assertFalse(CLOSED_LOOP._response_target_violation(28_000, 25, 3))
        self.assertTrue(CLOSED_LOOP._response_target_violation(28_001, 25, 3))
        self.assertTrue(CLOSED_LOOP._response_target_violation(21_999, 25, 3))
        self.assertFalse(CLOSED_LOOP._response_target_violation(50_000, None, 3))

    def test_invalid_and_conflicting_timing_arguments_are_rejected(self):
        for target, tolerance, delay in ((0, 3, 0), (25, 0, 0), (57, 3, 0), (25, 3, 1), (float('nan'), 3, 0)):
            with self.assertRaises(ValueError):
                CLOSED_LOOP._response_timing_contract(60, delay, target, tolerance)
        self.assertEqual(CLOSED_LOOP._response_timing_contract(60, 12),
                         {'mode': 'native_plus_added_delay', 'actuation_delay_ms': 12.0})
        self.assertNotEqual(CLOSED_LOOP._response_timing_contract(60, 0),
                            CLOSED_LOOP._response_timing_contract(60, 0, 25, 3))

    def test_plan_covers_every_model_once_per_repeat_and_command_is_frozen(self):
        self.assertEqual(CHECK._plan(), CHECK._plan())
        self.assertEqual(len(CHECK._plan()), 3)
        for order in CHECK._plan():
            self.assertEqual(sorted(order), [1, 2, 3, 4, 5, 6])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            command = CHECK._command(directory, [6, 2, 5, 1, 4, 3])
        self.assertEqual(command[command.index('--model-ids') + 1], '6,2,5,1,4,3')
        self.assertEqual(command[command.index('--target-response-ms') + 1], '25.0')
        self.assertEqual(command[command.index('--reference-profile') + 1], 'constant')
        self.assertNotIn('--auto-launch', command)

    def _fixture(self, root, changed_model_r_ms=None, corrupt_j=False):
        catalog = {i: dict(name=f'model{i}', test_normalized_mse=1.0/i) for i in range(1, 7)}
        for block in range(3):
            directory = root / f'repeat_{block:02d}'
            logs = directory / 'trial_logs'
            logs.mkdir(parents=True)
            trials = []
            for model_id in range(1, 7):
                key = f'm{model_id:02d}_S02_r00'
                elapsed_ms = changed_model_r_ms if model_id == 6 and changed_model_r_ms is not None else 26.0
                rows = []
                for cycle in range(21):
                    acquired = 1_000_000 + cycle * 60_000
                    rows.append(dict(model_id=model_id, scenario_id='S02', period_ms=60,
                        state_acquired_us=acquired, command_send_started_us=acquired+25_000,
                        command_dispatch_us=acquired+round(elapsed_ms*1000),
                        target_response_ms=25., response_tolerance_ms=3.,
                        command_source='fpga_fresh', clip_flags=[False]*4,
                        paper_state_error_squared_sum=1.0/(21*model_id)))
                (logs / f'{key}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                trials.append(dict(model_id=model_id, trial_key=key, samples=21, completed=True,
                    crossings=[{}]*5, error=None, collision=None, fallback_count=0,
                    deadline_miss_count=0, timeout_count=0, J_paper_style=model_id+(1 if corrupt_j else 0)))
            (directory/'fpga_closed_loop_summary.json').write_text(json.dumps(dict(trials=trials)))
        return catalog

    def test_report_checks_all_18_trials_and_sorts_by_audited_c(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = CHECK.summarize(root, self._fixture(root))
        self.assertTrue(report['timing_comparable'])
        self.assertEqual(report['clean_and_target_pass'], 18)
        self.assertEqual(report['trend']['model_ids_by_increasing_c'], [1,2,3,4,5,6])
        self.assertAlmostEqual(report['trend']['highest_vs_lowest_c_J_change_percent'], 500.0)

    def test_out_of_band_measurements_are_retained_and_block_trend_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = CHECK.summarize(root, self._fixture(root, changed_model_r_ms=31.0))
        self.assertEqual(report['episodes_recorded'], 18)
        self.assertFalse(report['timing_comparable'])
        self.assertIsNone(report['trend'])
        self.assertEqual(report['clean_and_target_pass'], 15)

    def test_different_model_latency_distributions_also_block_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = CHECK.summarize(root, self._fixture(root, changed_model_r_ms=27.5))
        self.assertEqual(report['clean_and_target_pass'], 18)
        self.assertFalse(report['timing_comparable'])
        self.assertEqual(report['observed_model_median_spread_ms'], 1.5)

    def test_incorrect_J_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = CHECK.summarize(root, self._fixture(root, corrupt_j=True))
        self.assertFalse(report['timing_comparable'])
        self.assertEqual(report['clean_fpga_episodes'], 0)


if __name__ == '__main__':
    unittest.main()
