"""Run the compiled native event/calendar code with a delayed mock RPC endpoint."""
import time

import numpy as np
import pytest

from config import ThrustCalibration
from controller_architecture_study.framework_native_io import LIBRARY,NativeCommandIO


pytestmark=pytest.mark.skipif(not LIBRARY.exists(),reason='build C++ runtime first')


def collect(io,kind,timeout=2.):
    end=time.perf_counter()+timeout
    while time.perf_counter()<end:
        result=getattr(io,kind)()
        if result:return result
        time.sleep(.001)
    raise AssertionError('native operation did not complete')


def plan(release,model='a',number=0):
    return dict(vehicle='Drone1',sample_id=f'Drone1:{number}',model=model,priority=0,
        planned_release_s=release,minimum_release_s=release,requested_period_s=.020,
        nominal_deadline_s=release+.020,deadline_s=release+.300,slot=number,status='planned')


def test_read_only_calendar_check_detects_publication_without_rewriting_it():
    io=NativeCommandIO(['Drone1'],ThrustCalibration(),.003,mock=True)
    try:
        origin=time.perf_counter()+.050
        row=plan(origin);io.schedule(row,origin+.100)
        assert io.calendar_matches({'Drone1':row})
        assert io.calendar_matches({'Drone1':row})
        time.sleep(.075)
        assert not io.calendar_matches({'Drone1':row})
        releases=io.drain_releases()
        assert releases[0]['sample_id']=='Drone1:0'
        assert releases[0]['planned_release_s']==origin
    finally:io.close()


def test_native_calendar_releases_while_rpc_and_python_consumer_are_blocked():
    io=NativeCommandIO(['Drone1'],ThrustCalibration(),.003,mock=True,mock_delay_s=.090,allow_late=True)
    try:
        origin=time.perf_counter()+.010
        io.schedule(plan(origin),origin+.100)
        io.sample(plan(origin))
        time.sleep(.115) # Python performs NO drains, plans or RPC management.
        rows=io.drain_releases()
        assert len(rows)==5
        assert [r['sample_id'] for r in rows]==[f'Drone1:{i}' for i in range(5)]
        assert np.allclose(np.diff([r['planned_release_s'] for r in rows]),.020,atol=1e-8)
        assert all(r['native_release_s']>=r['planned_release_s'] for r in rows)
        assert max(r['native_release_s'] for r in rows)<time.perf_counter()
        assert len(io.drain_samples())==1
    finally:io.close()


def test_mode_update_cannot_rewrite_already_released_events_or_reuse_ids():
    io=NativeCommandIO(['Drone1'],ThrustCalibration(),.003,mock=True)
    try:
        origin=time.perf_counter()+.010
        io.schedule(plan(origin,'old'),origin+.150)
        time.sleep(.055)
        io.schedule(plan(time.perf_counter()+.010,'new'),origin+.150)
        time.sleep(.040)
        rows=io.drain_releases()
        assert len(rows)>=3
        models=[r['model'] for r in rows]
        assert models[0]=='old' and models[-1]=='new'
        first_new=models.index('new')
        assert all(m=='old' for m in models[:first_new])
        assert len({r['sample_id'] for r in rows})==len(rows)
        assert all(r['deadline_s']==pytest.approx(origin+.300) for r in rows[:first_new])
    finally:io.close()


def test_native_command_mapping_deadline_and_postcommand_sample():
    io=NativeCommandIO(['Drone1'],ThrustCalibration(),.003,mock=True,allow_late=False)
    try:
        row=plan(time.perf_counter());io.send(np.array([30.,4.,-4.,3.]),row)
        packet,done=collect(io,'drain')[0]
        assert done['control']==[20.,3.,-3.,2.]
        assert done['airsim_command']['pitch_rate']==3.
        assert done['airsim_command']['yaw_rate']==-2.
        # Check the real Python client's WIRE arguments, not only our adapter's
        # public API arguments. The Python client itself flips pitch/yaw again.
        from controller_architecture_study.fly_convex_policy import load_airsim_client
        from airsim_interface import require_airsim,AirSimDrone
        from control_adapter import send_control
        from unittest.mock import Mock
        load_airsim_client()
        client=require_airsim().MultirotorClient.__new__(require_airsim().MultirotorClient)
        client.client=Mock()
        send_control(AirSimDrone(client,'Drone1'),done['control'],ThrustCalibration(),duration=1.)
        args=client.client.call_async.call_args.args
        assert args[0]=='moveByAngleRatesThrottle'
        assert done['native_rpc_arguments']==pytest.approx(args[1:5])
        assert packet[2]>=done['rpc_return_s'] and packet[0].quaternion[0]==1.
        row=plan(time.perf_counter(),number=1);row['deadline_s']=time.perf_counter()-1.
        io.send(np.array([9.81,0.,0.,0.]),row)
        packet,done=collect(io,'drain')[0]
        assert packet is None and done['status']=='expired_not_applied' and 'rpc_return_s' not in done
    finally:io.close()


def test_native_group_update_is_all_or_none_when_a_release_races_the_forecast():
    io=NativeCommandIO(['Drone1','Drone2'],ThrustCalibration(),.003,mock=True)
    try:
        origin=time.perf_counter()+.020
        first={'Drone1':plan(origin), 'Drone2':{**plan(origin+.100),'vehicle':'Drone2','sample_id':'Drone2:0'}}
        assert io.schedule_batch(first,origin+.200)
        time.sleep(.045)
        # Drone1's zero-th job has already been published; Drone2's has not.
        changed={v:{**r,'model':'new','planned_release_s':origin+.090} for v,r in first.items()}
        assert not io.schedule_batch(changed,origin+.200)
        assert io.scheduled['Drone2']==io.plan_key(first['Drone2'])
        time.sleep(.105)
        rows=io.drain_releases()
        assert any(r['vehicle']=='Drone2' for r in rows)
        assert all(r['model']=='a' for r in rows)
    finally:io.close()


def test_native_group_checks_unchanged_participants_for_a_release_race_too():
    io=NativeCommandIO(['Drone1','Drone2'],ThrustCalibration(),.003,mock=True)
    try:
        origin=time.perf_counter()+.020
        first={'Drone1':plan(origin), 'Drone2':{**plan(origin+.100),'vehicle':'Drone2','sample_id':'Drone2:0'}}
        assert io.schedule_batch(first,origin+.200)
        time.sleep(.045)
        # The forecast also relied on Drone1, although only Drone2 changes.
        changed={**first,'Drone2':{**first['Drone2'],'model':'new'}}
        assert not io.schedule_batch(changed,origin+.200)
        assert io.scheduled['Drone2']==io.plan_key(first['Drone2'])
    finally:io.close()


def test_guarded_dry_run_has_no_side_effects_and_detects_same_id_new_version():
    io=NativeCommandIO(['Drone1'],ThrustCalibration(),.003,mock=True)
    try:
        origin=time.perf_counter()+.5;row=plan(origin);io.schedule(row,origin+1.)
        before=io.version;basis=io.calendar_basis({'Drone1':row})
        changed={'Drone1':{**row,'model':'b','planned_release_s':origin+.02}}
        assert io.guarded_batch(basis,changed,{'Drone1':0},origin+1.,origin)['passed']
        assert io.version==before and io.calendar_matches({'Drone1':row})
        io.schedule({**row,'model':'new'},origin+1.)
        check=io.guarded_batch(basis,changed,{'Drone1':0},origin+1.,origin)
        assert not check['passed'] and check['reason']=='calendar_changed'
    finally:io.close()


def test_guarded_group_rejects_changed_actual_control_even_before_python_drains():
    io=NativeCommandIO(['Drone1','Drone2'],ThrustCalibration(),.003,mock=True,mock_delay_s=.03)
    try:
        origin=time.perf_counter()+.5
        rows={'Drone1':plan(origin),'Drone2':{**plan(origin),'vehicle':'Drone2','sample_id':'Drone2:0'}}
        assert io.schedule_batch(rows,origin+1.)
        basis=io.calendar_basis(rows);tokens={'Drone1':0,'Drone2':0}
        command=plan(time.perf_counter());io.send(np.array([9.8,0.,0.,0.]),command)
        assert not io.guarded_batch(basis,rows,tokens,origin+1.,origin)['passed']
        time.sleep(.06)  # Native ack changes the context, without a Python pump.
        assert io.guarded_batch(basis,rows,tokens,origin+1.,origin)['reason']=='control_changed_or_inflight'
        collect(io,'drain');tokens['Drone1']=command['control_ack_token']
        assert io.guarded_batch(basis,rows,tokens,origin+1.,origin)['passed']
        assert not io.guarded_batch(basis,rows,tokens,origin+1.,time.perf_counter()-1.)['passed']
        proposal={v:{**r,'model':'b','planned_release_s':origin+.02} for v,r in rows.items()}
        assert io.guarded_batch(basis,proposal,tokens,origin+1.,origin,commit=True)['installed']
        assert io.calendar_matches(proposal)
        assert all(io.templates[io.scheduled_versions[v]]['model']=='b' for v in rows)
    finally:io.close()
