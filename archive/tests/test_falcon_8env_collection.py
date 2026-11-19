from collections import Counter
from copy import deepcopy
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

HERE = Path(__file__).resolve().parents[1] / 'experiments/falcon_svr_8env_v1'
sys.path.insert(0, str(HERE))
import experiment as EXP
import collect as COLLECT
import export_and_plot as EXPORT
from run_fpga_closed_loop import _scenario_gates
import run_fpga_closed_loop as LOOP
from environment import generate_gate_sequence
from airsim_interface import RawDroneState


class EightEnvTests(unittest.TestCase):
    def setUp(self):
        self.config, self.benchmark, self.catalog = EXP.load_experiment(HERE / 'collection_config.json')
        self.plan = EXP.make_plan(self.config)

    def record(self, error=None, collision=None, completed=True, fallback=0):
        job = self.plan[0]
        scenario = next(s for s in EXP.scenarios(self.config) if s['id'] == job['scenario_id'])
        return dict(job=job, scenario=scenario, raw_log='attempt_0000.jsonl', wall_seconds=10.,
            trial=dict(completed=completed, error=error, collision=collision, samples=100,
                fallback_count=fallback, timeout_count=0, deadline_miss_count=0,
                J_paper_style=.003, measured_R_max_ms=26., measured_R_p95_ms=25.8,
                measured_R_p99_ms=25.9, state_to_command_dispatch_us=[25500]*100,
                crossings=[{}]* (5 if completed else 1)))

    def model(self, record):
        return next(m for m in self.catalog['models'] if m['model_id'] == record['job']['model_id'])

    def test_full_plan_is_dense_balanced_unique_and_reproducible(self):
        self.assertEqual(len(self.plan), 4608)
        self.assertEqual(self.plan, EXP.make_plan(self.config))
        self.assertEqual(len({j['id'] for j in self.plan}), 4608)
        self.assertEqual(set(Counter(j['environment_id'] for j in self.plan).values()), {576})
        for env in self.config['environments']:
            points = {(j['model_id'],j['period_ms'],j['target_R_ms']) for j in self.plan if j['environment_id']==env['id']}
            self.assertEqual(len(points), 288)
        self.assertEqual(len({(j['scenario_id'], j['model_id']) for j in self.plan[:96]}), 96)

    def test_geometry_is_explicit_clear_of_ground_and_has_no_reverse_reference_segment(self):
        for scenario in EXP.scenarios(self.config):
            centers = np.asarray(scenario['gate_centers'])
            dx = np.diff(centers[:,0])
            self.assertTrue(np.all(dx > 0))
            np.testing.assert_allclose(np.abs(np.diff(centers[:,1:],axis=0)).sum(axis=1),scenario['epsilon'])
            overrides = scenario['gate_overrides']
            self.assertLess(overrides['approach_distance']+overrides['exit_distance'],dx.min())
            self.assertLess(overrides['completion_radius'],overrides['exit_distance'])
            self.assertTrue(np.all(centers[:,2] + 2.075 < 0))
            gates = _scenario_gates(scenario,self.benchmark['gate'])
            np.testing.assert_allclose([g.center for g in gates],centers)

    def test_original_environment_contract_is_not_relaxed(self):
        with self.assertRaises(ValueError):
            generate_gate_sequence(6.8)

    def test_success_preserves_original_J(self):
        record = self.record()
        row = EXP.outcome_row(record,self.model(record))
        self.assertTrue(row['valid_for_success_svr'])
        self.assertEqual(row['J_paper_style_success'],.003)
        self.assertEqual(row['J_failure_zero'],.003)
        self.assertEqual(row['R_mean_ms'],25.5)

    def test_collision_zero_is_separate_from_success_J(self):
        record = self.record(error='RuntimeError: AirSim collision with FalconGate_2_left',collision='FalconGate_2_left',completed=False)
        row = EXP.outcome_row(record,self.model(record))
        self.assertFalse(row['valid_for_success_svr'])
        self.assertTrue(row['valid_for_failure_zero_svr'])
        self.assertIsNone(row['J_paper_style_success'])
        self.assertEqual(row['J_failure_zero'],0)
        self.assertEqual(row['partial_J_diagnostic'],.003)

    def test_RPC_timeout_is_not_mislabeled_as_mission_timeout(self):
        record = self.record(error='TimeoutError: Request timed out',completed=False)
        row = EXP.outcome_row(record,self.model(record))
        self.assertEqual(row['outcome'],'infrastructure_error')
        self.assertFalse(row['valid_for_failure_zero_svr'])
        self.assertIsNone(row['J_failure_zero'])
        for error in ('TimeoutError: episode exceeded 25.0s','TimeoutError: gate 2 was not crossed within timeout'):
            record['trial']['error'] = error
            self.assertEqual(EXP.outcome_row(record,self.model(record))['J_failure_zero'],0)

    def test_mixed_teacher_or_tardy_control_is_not_a_pure_FPGA_target(self):
        record = self.record(fallback=1)
        self.assertFalse(EXP.outcome_row(record,self.model(record))['valid_for_failure_zero_svr'])
        record['trial']['fallback_count'] = 0
        record['trial']['deadline_miss_count'] = 1
        self.assertFalse(EXP.outcome_row(record,self.model(record))['valid_for_success_svr'])

    def test_rectangular_grid_rejects_infeasible_R_T_combinations(self):
        config = deepcopy(self.config)
        config['target_responses_ms'] = [60]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                EXP.load_experiment(path)

    def test_completed_collision_is_not_reflown_on_resume(self):
        record = self.record(error='RuntimeError: collision',collision='Gate2',completed=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            EXP.write_json(root/'trials'/record['job']['id']/'result.json',record)
            with patch.object(COLLECT,'require_airsim',side_effect=AssertionError('No hardware calls expected')):
                result=COLLECT.collect(root,self.config,self.benchmark,self.catalog,[record['job']],None)
        self.assertEqual(result['recorded'],1)
        self.assertEqual(result['collisions'],1)
        self.assertEqual(result['status'],'complete')

    def test_interrupted_attempt_is_kept_and_next_attempt_gets_new_name(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            first=directory/'attempt_0000.jsonl'
            first.write_text('partial data')
            self.assertEqual(COLLECT.next_attempt(directory),1)
            self.assertEqual(first.read_text(),'partial data')

    def test_resume_detects_configuration_and_frozen_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            EXP.freeze(root,HERE/'collection_config.json')
            EXP.freeze(root,HERE/'collection_config.json')
            path=root/'initial_states.json'
            path.write_text('[]')
            with self.assertRaisesRegex(ValueError,'Frozen experiment file changed'):
                EXP.freeze(root,HERE/'collection_config.json')

    def test_native_profiling_does_not_fly_and_records_all_six_models(self):
        reply=SimpleNamespace(fresh=True,rtt_us=16000,dpu_time_us=12000)
        udp=MagicMock()
        udp.__enter__.return_value=udp
        udp.warmup.return_value=[reply]*5
        udp.infer_normalized.return_value=reply
        udp.stats={'requests':630,'fresh_responses':630}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(COLLECT,'Falcon12FpgaUdpClient',return_value=udp), patch.object(COLLECT,'require_airsim',side_effect=AssertionError('Must not fly')):
                report=COLLECT.profile(root,self.config,self.catalog)
                cached=COLLECT.profile(root,self.config,self.catalog)
        self.assertEqual(len(report['models']),6)
        self.assertEqual(udp.infer_normalized.call_count,600)
        self.assertEqual(cached,report)
        self.assertEqual(report['empirical_target_floor_ms'],19.)

    def test_profile_rejects_a_target_below_empirical_latency_floor(self):
        reply=SimpleNamespace(fresh=True,rtt_us=24000,dpu_time_us=12000)
        udp=MagicMock()
        udp.__enter__.return_value=udp
        udp.warmup.return_value=[reply]*5
        udp.infer_normalized.return_value=reply
        udp.stats={}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(COLLECT,'Falcon12FpgaUdpClient',return_value=udp):
                with self.assertRaisesRegex(RuntimeError,'too close'):
                    COLLECT.profile(root,self.config,self.catalog)
            self.assertFalse((root/'latency_profile.json').exists())
            self.assertTrue((root/'latency_profile_rejected.json').exists())

    def test_export_retains_failure_and_censors_infrastructure(self):
        good=self.record(error='GatePassageError: outside opening',completed=False)
        bad=deepcopy(good)
        bad['raw_log']='attempt_0000.jsonl'
        bad['trial']['error']='TimeoutError: Request timed out'
        good['raw_log']='attempt_0001.jsonl'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            EXP.freeze(root,HERE/'collection_config.json')
            trial_dir=root/'trials'/good['job']['id']
            EXP.write_json(trial_dir/'attempt_0000.summary.json',bad)
            EXP.write_json(trial_dir/'attempt_0001.summary.json',good)
            EXP.write_json(trial_dir/'result.json',good)
            rows=EXPORT.export_dataset(root)
            manifest=EXP.read_json(root/'datasets/dataset_manifest.json')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['J_failure_zero'],0)
        self.assertEqual(manifest['censored_attempts'],1)

    def test_collision_is_preserved_even_if_cleanup_reset_fails(self):
        client=MagicMock()
        client.reset.side_effect=[None,RuntimeError('cleanup failed')]
        drone=MagicMock()
        drone.get_raw_state.return_value=RawDroneState(position=np.array([0.,0.,-3.]),
            quaternion=np.array([1.,0.,0.,0.]), velocity=np.zeros(3), acceleration=np.zeros(3),
            angular_velocity=np.zeros(3))
        scene=MagicMock()
        udp=MagicMock()
        udp.warmup.return_value=[SimpleNamespace(fresh=True)]*5
        scenario=EXP.scenarios(self.config)[0]
        with patch.object(LOOP,'AirSimDrone',return_value=drone), patch.object(LOOP,'spawn_gate_sequence',return_value=scene), \
             patch.object(LOOP,'vehicle_local_to_world_offset',return_value=np.zeros(3)), \
             patch.object(LOOP,'cleanup_generated_gates'), patch.object(LOOP,'_pose'), \
             patch.object(LOOP.time,'sleep'), patch.object(LOOP,'normalized_input',return_value=np.zeros((1,12))), \
             patch.object(LOOP,'_check_collision',return_value=(True,'FalconGate_1_left')):
            result=LOOP._run_trial(client=client,airsim=MagicMock(),scenario=scenario,repeat_index=0,
                benchmark=self.benchmark,model=dict(model_id=1,model_name='depth10_width512',normalization={}),
                udp_client=udp,shadow=dict(warmup_requests_per_model=5),period_s=.06,actuation_delay_s=0,
                calibration=None,vehicle_name='Drone1',initial_states={scenario['initial_state_id']:object()},
                output_handle=io.StringIO(),experiment_id='unit_fixture',keep_gates=False,reset_after_trial=True)
        self.assertEqual(result['collision'],'FalconGate_1_left',result)
        self.assertIn('cleanup failed',result['cleanup_error'])
        self.assertFalse(result['completed'])
        drone.land.assert_not_called()
        scene.destroy.assert_called_once()

    def test_collection_limit_then_resume_keeps_failed_trial_exactly_once(self):
        jobs=self.plan[:2]
        scenario_map={s['id']:s for s in EXP.scenarios(self.config)}
        fake_client=MagicMock()
        fake_client.listVehicles.return_value=['Drone1']
        fake_client.getSettingsString.return_value='{}'
        fake_client.getClientVersion.return_value=1
        fake_client.getServerVersion.return_value=1
        api=SimpleNamespace(MultirotorClient=MagicMock(return_value=fake_client))
        udp=MagicMock()
        udp.__enter__.return_value=udp
        states=[SimpleNamespace(initial_state_id=i) for i in self.config['initial_state_ids']]
        table=[dict(model_id=m['model_id']) for m in self.catalog['models']]
        first=self.record(error='RuntimeError: collision',collision='Gate2',completed=False)['trial']
        second=self.record()['trial']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,frozen_benchmark,_,_=EXP.freeze(root,HERE/'collection_config.json')
            with patch.object(COLLECT,'profile'), patch.object(COLLECT,'require_airsim',return_value=api), \
                 patch.object(COLLECT,'_model_table',return_value=table), \
                 patch.object(COLLECT,'load_initial_states',return_value=states), \
                 patch.object(COLLECT,'Falcon12FpgaUdpClient',return_value=udp), \
                 patch.object(COLLECT,'_run_trial',side_effect=[first,second]) as run:
                p=COLLECT.collect(root,self.config,frozen_benchmark,self.catalog,jobs,limit=1)
                self.assertEqual(p['recorded'],1)
                self.assertEqual(p['collisions'],1)
                p=COLLECT.collect(root,self.config,frozen_benchmark,self.catalog,jobs,limit=None)
                self.assertEqual(p['recorded'],2)
                self.assertEqual(p['collisions'],1)
                COLLECT.collect(root,self.config,frozen_benchmark,self.catalog,jobs,limit=None)
                self.assertEqual(run.call_count,2)
            self.assertEqual(EXP.read_json(root/'trials'/jobs[0]['id']/'result.json')['scenario'],scenario_map[jobs[0]['scenario_id']])


if __name__=='__main__':
    unittest.main()
