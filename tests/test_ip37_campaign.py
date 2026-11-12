"""Campaign isolation, selection fairness and independent evaluation contracts."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study.ip37_campaign import neighbors, best_static, paired, run_trial, main
from controller_architecture_study.ip37_evaluation import observation, Evaluator
from controller_architecture_study.ip37_joint_runtime import JointRuntime
from controller_architecture_study.ip37_safety_cases import make_case
from controller_architecture_study.decision_dataset import atomic_json


class CampaignTests(unittest.TestCase):
    def test_role_routes_are_mixed_and_start_at_same_reference_progress(self):
        case = make_case(0, 1.7, vehicles=12)
        self.assertEqual(len(case['layouts']), 12)
        self.assertEqual([case['roles'][f'Drone{i}']['role'] for i in range(1, 5)],
                         ['early', 'middle', 'late', 'smooth'])
        self.assertFalse(case['external_disturbance'])
        for i in range(1, 13):
            self.assertEqual(case['layouts'][f'Drone{i}']['segments'][0]['start_time'], 0.)
        times = [case['roles'][f'Drone{i}']['turn_gate_times_s'][0] for i in (1, 2, 3)]
        np.testing.assert_allclose(np.diff(times), [6., 6.])

    def test_greedy_edits_one_role_and_keeps_model_fixed(self):
        p = dict(periods={'a': 60., 'b': 60., 'c': 60.}, order='edf')
        roles = {'a': {'role': 'early'}, 'b': {'role': 'early'}, 'c': {'role': 'smooth'}}
        options = list(neighbors(p, roles))
        self.assertEqual(len(options), 5)
        self.assertEqual(options[0]['periods'], {'a': 50., 'b': 50., 'c': 60.})
        self.assertEqual(options[-1]['order'], 'fifo')
        self.assertEqual(p['periods']['a'], 60.)

    def test_selection_uses_complete_J_and_timing_not_just_zero_collisions(self):
        partial = dict(comparable=False, J_total=1000., timing=True, requests=1, collisions=0)
        fast = dict(comparable=True, J_total=9., timing=False, requests=100, collisions=0)
        timely = dict(comparable=True, J_total=8., timing=True, requests=50, collisions=1)
        self.assertIs(best_static([partial, fast, timely]), timely)
        self.assertIs(best_static([partial, fast]), fast)
        self.assertIsNone(best_static([partial]))

    def test_pairing_does_not_call_an_aborted_loop_rescued(self):
        common = dict(case_sha256='a', wire_tag_hex='b', profile={'order': 'edf'}, comparable=True,
                      J_total=1., collisions=1, first_install_t_s=None, first_contact_t_s=6.)
        a = dict(common, contacts={'Drone1': {}}, passed_vehicles=['Drone2'])
        b = dict(common, contacts={}, passed_vehicles=['Drone1'], J_total=2., first_install_t_s=4.)
        result = paired(a, b)
        self.assertEqual(result['rescued_vehicles'], ['Drone1'])
        self.assertEqual(result['newly_failed_vehicles'], ['Drone2'])
        self.assertTrue(result['installed_before_static_contact'])
        with self.assertRaises(ValueError):
            paired(a, dict(b, case_sha256='wrong'))

    def test_completed_trial_resume_never_launches_a_process(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            case, bundle = root/'case.json', root/'bundle'
            bundle.mkdir()
            atomic_json(case, {'case': 1})
            atomic_json(bundle/'manifest.json', {'model': 11})
            args = SimpleNamespace(bundle=bundle, evaluation_ms=50.)
            profile = {'periods': {'Drone1': 80.}, 'order': 'edf'}
            from controller_architecture_study.decision_dataset import sha256
            metadata = dict(case_sha256=sha256(case), profile=profile, policy='fixed',
                source_sha256={}, bundle_sha256=sha256(bundle/'manifest.json'), evaluation_ms=50., model_id=11)
            folder = root/'trial'
            atomic_json(folder/'trial.json', metadata)
            atomic_json(folder/'trial_summary.json', {'saved': True})
            with patch('subprocess.run') as launch:
                self.assertEqual(run_trial(args, case, profile, 'fixed', folder, {}), {'saved': True})
                launch.assert_not_called()
            with self.assertRaises(ValueError):
                run_trial(args, case, profile, 'adaptive', folder, {})

    def test_full_search_freezes_before_alternating_evaluation(self):
        calls = []
        def fake_trial(args, case, profile, policy, folder, sources):
            from copy import deepcopy
            calls.append((policy, deepcopy(profile)))
            j = 100-sum(abs(t-60) for t in profile['periods'].values())/100
            return dict(directory=str(folder), profile=deepcopy(profile), policy=policy,
                comparable=True, J_total=j, timing=True, requests=100, flight=True,
                collisions=0, contacts={}, passed_vehicles=list(profile['periods']), completed_loops=4,
                installed=1 if policy == 'adaptive' else 0, period_updates=0,
                response_ms={'p99': 5., 'max_observed': 6.}, command_gaps_ms={},
                case_sha256='case', wire_tag_hex='wire', first_contact_t_s=None, first_install_t_s=None)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            bundle = root/'bundle'
            atomic_json(bundle/'manifest.json', {'model': 11})
            argv = ['campaign', '--bundle', str(bundle), '--output', str(root/'out'),
                    '--drones', '4', '--widths', '2.1', '--search-budget', '4', '--repeats', '2']
            with patch('sys.argv', argv), patch('controller_architecture_study.ip37_campaign.audit',
                    return_value={'passed': True}), patch(
                    'controller_architecture_study.ip37_campaign.run_trial', side_effect=fake_trial):
                main()
            self.assertEqual([p for p, _ in calls], ['fixed']*4+['fixed', 'adaptive', 'adaptive', 'fixed'])
            self.assertTrue(all(set(p['periods'].values()) == {60.} for _, p in calls[4:]))
            self.assertTrue((root/'out'/'pairs.csv').exists())


class EvaluationTests(unittest.TestCase):
    def test_quaternion_sign_does_not_create_false_state_cost(self):
        raw = SimpleNamespace(position=np.array([1., 0., 0.]), velocity=np.zeros(3),
                              quaternion=np.array([-1., 0., 0., 0.]), timestamp_ns=0)
        ref = SimpleNamespace(position=np.zeros(3), velocity=np.zeros(3), orientation=np.array([1., 0., 0., 0.]))
        self.assertEqual(observation(raw, ref, 0., 10.)['state_cost'], 1.)

    def evaluator(self):
        e = object.__new__(Evaluator)
        e.period, e.duration, e.error, e.rpc_ms = .05, .5, None, []
        e.trace = [dict(t=float(t), sim_ns=int(t*1e9), state_cost=1.) for t in np.arange(0., .501, .05)]
        return e

    def test_incomplete_or_long_gap_score_is_not_rankable(self):
        e = self.evaluator()
        self.assertTrue(e.report()['valid'])
        e.trace.pop()
        self.assertFalse(e.report()['valid'])
        e = self.evaluator()
        e.trace = [e.trace[0], e.trace[-1]]
        self.assertFalse(e.report()['valid'])
        e = self.evaluator()
        e.duration, e.trace = .1, e.trace[:3]
        self.assertFalse(e.report()['valid'])


class InitialPeriodsTests(unittest.TestCase):
    def test_per_vehicle_periods_remain_calendar_based_when_busy(self):
        rt = JointRuntime(['a', 'b'], 10., 80., [40., 80.], 11, 2.,
                          initial_periods={'a': 40., 'b': 80.}, adaptive=False)
        first, = rt.release_due(10.)
        next_rows = rt.release_due(10.04001)
        self.assertEqual(first['deadline_s'], 10.04)
        self.assertEqual(rt.rows[1]['status'], 'dropped_busy_io')
        self.assertEqual([r['vehicle'] for r in next_rows], ['b'])
        self.assertAlmostEqual(rt.loops['a']['next'], 10.08)

    def test_fixed_EDF_cannot_use_state_feedback(self):
        rt = JointRuntime(['a', 'b'], 10., 50., [50.], 11, 2., adaptive=False)
        jobs = rt.release_due(10.03)
        with patch('controller_architecture_study.ip37_joint_runtime.time.perf_counter', return_value=10.03):
            for job in jobs:
                rt.enqueue(job, np.zeros(237))
        jobs[1]['safety_deadline_s'] = 10.031
        self.assertEqual(rt.take(10.03)['vehicle'], 'a')
        rt.adaptive = True
        rt.active = None
        rt.queue = jobs
        self.assertEqual(rt.take(10.03)['vehicle'], 'b')


if __name__ == '__main__':
    unittest.main()
