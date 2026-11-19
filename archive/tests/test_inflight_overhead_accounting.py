import pytest

from controller_architecture_study.diagnose_inflight_overhead import response_parts, cycle_parts


def test_response_accounting_separates_fc_work_from_wait_and_closes_exactly():
    times={"sample_submitted_s":0.,"sample_helper_received_s":1.,"sample_io_started_s":2.,
        "sample_io_finished_s":4.,"sample_response_received_s":7.,"feature_start_s":9.,
        "feature_done_s":10.,"enqueue_s":10.1,"worker_enqueue_s":10.2,"first_service_s":10.3,
        "ready_s":14.5,"io_submitted_s":15.,"command_helper_received_s":16.,"io_started_s":17.,"rpc_return_s":18.}
    row={k:v/1000 for k,v in times.items()}
    row["compute_ms"]=3.5
    parts=response_parts(row)
    assert sum(parts.values())==pytest.approx(18.)
    assert parts["cpu_dnn_compute"]==3.5
    assert parts["between_segments"]==pytest.approx(.7)
    assert parts["sample_completion_delivery"]==pytest.approx(3.)
    row["compute_ms"]=8.  # It cannot be larger than the entire segment span.
    with pytest.raises(ValueError):
        response_parts(row)


def test_cycle_distinguishes_waiting_for_calendar_from_dispatching_late():
    previous=dict(rpc_return_s=1.,io_finished_s=1.002,io_response_received_s=1.009)
    following=dict(planned_release_s=1.020,job_released_s=1.025,sample_submitted_s=1.026,rpc_return_s=1.044)
    parts=cycle_parts(previous,following)
    assert sum(parts.values())==pytest.approx(44.)
    assert parts["calendar_or_event_wait"]==pytest.approx(11.)
    assert parts["release_lateness_after_available"]==pytest.approx(5.)
    following["planned_release_s"]=1.1
    with pytest.raises(ValueError):
        cycle_parts(previous,following)
