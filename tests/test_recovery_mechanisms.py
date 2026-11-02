from controller_architecture_study.probe_recovery_mechanisms import select_static
from controller_architecture_study.probe_seven_continuous_timing import control_deadline


def test_ablation_removes_physical_bound_but_keeps_held_control_update_cap():
    budget={"deadline_s":1.030}
    assert control_deadline(1.060,budget,1.,True)==1.030
    assert control_deadline(1.060,budget,1.,False)==1.060
    assert control_deadline(1.200,budget,1.,False)==1.080
    assert budget=={"deadline_s":1.030}


def test_static_search_rejects_missing_updates_even_if_completed_jobs_look_good():
    def row(score,misses,success=True):
        return dict(report=dict(success=success,total_J=score),normal_metrics=dict(effective_misses=misses))
    valid=row(1.,0)
    assert select_static([row(10.,1),valid,row(20.,0,False)]) is valid
    assert select_static([row(10.,1),row(None,0,False)]) is None
