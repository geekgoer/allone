from unittest.mock import patch

from controller_architecture_study.framework_runtime import simulate_pipeline
from controller_architecture_study.heterogeneous_deadlines import admit
from controller_architecture_study.heterogeneous_load_guard import allocate


def test_helping_one_loop_can_break_another_despite_low_compute_load():
    profile={'m':{'segment_estimates_s':[.002,.002,.002]}}
    costs=dict(input_s=.001,output_s=.001,completion_guard_s=0.)
    old=dict(vehicle='b',sample_id='b:0',model='m',priority=1,
             planned_release_s=0.,deadline_s=.012,stage='compute',ingress_remaining_s=0.)
    urgent=dict(vehicle='a',sample_id='a:0',model='m',priority=0,
                planned_release_s=0.,minimum_release_s=0.,deadline_s=.010)
    periods,guard=allocate({'a':.030,'b':.030},{'a':1.,'b':0.},
                          {'a':'m','b':'m'},{'m':.006})
    assert guard['feasible'] and abs(guard['load']-.4)<1e-12
    assert simulate_pipeline(0.,[],[old],profile,costs)[0]
    assert simulate_pipeline(0.,[urgent],[],profile,costs)[0]
    good,finish,_=simulate_pipeline(0.,[urgent],[old],profile,costs)
    assert not good and finish['a:0']<=urgent['deadline_s']
    assert 'b:0' not in finish or finish['b:0']>old['deadline_s']
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        proposal,info=admit(0.,{'a':urgent},[old],profile,costs)
    assert proposal is None and not info['feasible']
    assert urgent['deadline_s']==.010 and old['deadline_s']==.012


def test_short_remaining_compute_does_not_remove_existing_io_deadline():
    profile={'m':{'segment_estimates_s':[.0001]}}
    costs=dict(input_s=.001,output_s=.001,completion_guard_s=0.,io_s=.005,command_s=.004)
    pending=dict(vehicle='b',sample_id='b:1',model='m',priority=1,planned_release_s=0.,
                 deadline_s=.003,stage='io',io_remaining_s=.005,command_remaining_s=.004)
    with patch('controller_architecture_study.heterogeneous_deadlines.time.perf_counter',return_value=0.):
        proposal,info=admit(0.,{},[pending],profile,costs)
    assert proposal is None and not info['feasible']
