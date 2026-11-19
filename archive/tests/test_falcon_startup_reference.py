import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/paper_track_p_12d_5ms_v1/fpga_shadow_historical5ms'
sys.path.insert(0,str(HERE))
from startup_reference import StartupPolylineReference, SmoothStartupGateRoute, reference_contract
from environment import generate_gate_sequence


class StartupReferenceTests(unittest.TestCase):
    def setUp(self):
        self.initial=np.array([.1,-.08,.06])
        self.reference=StartupPolylineReference([[0,0,-2],[8,0,-2],[11,0,-2],[21,0,-2]],self.initial)

    def test_initial_velocity_is_measured_vector_without_jump(self):
        r=self.reference.at(0)
        np.testing.assert_allclose(r.position,[0,0,-2])
        np.testing.assert_allclose(r.velocity,self.initial)
        np.testing.assert_allclose(r.acceleration,0)

    def test_ramp_joins_original_path_with_continuous_position_velocity_acceleration(self):
        before=self.reference.at(2-1e-7)
        after=self.reference.at(2+1e-7)
        np.testing.assert_allclose(before.position,after.position,atol=2e-6)
        np.testing.assert_allclose(before.velocity,after.velocity,atol=1e-6)
        np.testing.assert_allclose(before.acceleration,after.acceleration,atol=1e-5)
        np.testing.assert_allclose(self.reference.at(2).velocity,[5,0,0])
        np.testing.assert_allclose(self.reference.at(2).position,[5.1,0,-2])

    def test_position_velocity_and_acceleration_are_consistent_derivatives(self):
        t,h=.73,1e-5
        a,b=self.reference.at(t-h),self.reference.at(t+h)
        np.testing.assert_allclose((b.position-a.position)/(2*h),self.reference.at(t).velocity,atol=1e-8)
        np.testing.assert_allclose((b.velocity-a.velocity)/(2*h),self.reference.at(t).acceleration,atol=1e-8)

    def test_same_physical_time_reference_does_not_depend_on_sampling_period(self):
        route=SmoothStartupGateRoute(generate_gate_sequence(.8,seed=917,gate_count=5),[0,0,-2],
            initial_velocity=self.initial,startup_duration=2,approach_distance=2,exit_distance=1,
            reference_speed=5,start_time=10)
        direct=route.reference_at_time(10.6)
        fine=route.reference_window(10.57,dt=.03,horizon=1)[0]
        coarse=route.reference_window(10.5,dt=.1,horizon=1)[0]
        np.testing.assert_allclose(direct.position,fine.position)
        np.testing.assert_allclose(direct.velocity,coarse.velocity)

    def test_invalid_ramp_cannot_cross_first_corner_before_cruise(self):
        with self.assertRaises(ValueError):
            StartupPolylineReference([[0,0,0],[1,0,0]],np.zeros(3),duration=2)

    def test_profile_versions_keep_legacy_and_new_data_distinct(self):
        self.assertNotEqual(reference_contract(),reference_contract('smooth-start',2))
        with self.assertRaises(ValueError):
            reference_contract('smooth-start',float('nan'))

    def test_dataset_builder_rejects_mixed_reference_profiles_before_writing(self):
        spec=importlib.util.spec_from_file_location('startup_dataset_builder',HERE/'build_falcon_svr_dataset.py')
        builder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            manifest=temp/'models.json'
            manifest.write_text(json.dumps({'models':[dict(model_id=1,name='depth10_width512',
                test_physical_mse=.1,test_normalized_mse=.2,parameters=10)]}))
            trial=dict(model_id=1,scenario_id='S02',repeat_index=0)
            def summaries(path):
                profile='smooth-start' if path.name=='new' else 'constant'
                return {},dict(experiment_id=path.name,reference_profile=reference_contract(profile,2)),[trial]
            args=['build','--input-dir',str(temp/'old'),'--input-dir',str(temp/'new'),
                  '--model-manifest',str(manifest),'--output-dir',str(temp/'dataset')]
            with patch.object(sys,'argv',args),patch.object(builder,'_summary_trials',side_effect=summaries):
                with self.assertRaisesRegex(ValueError,'cannot combine different reference profiles'):
                    builder.main()
            self.assertFalse((temp/'dataset').exists())


if __name__=='__main__':
    unittest.main()
