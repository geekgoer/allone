from controller_architecture_study.audit_ip37_command_gaps import inspect


def test_gap_counts_submitted_commands_and_only_bracketed_missing_jobs():
    rows = [dict(vehicle='Drone1', cycle=k, release_s=k*.05, period_ms=50, status='dropped_busy_io')
            for k in range(6)]
    rows[0].update(status='command_return_late', command_submitted_s=.02, command_return_s=.06)
    rows[2].update(queue_s=.106)
    rows[3].update(queue_s=.154, board_start_s=.155, board_done_s=.157)
    rows[4].update(status='delivered', command_submitted_s=.22, command_return_s=.23)
    r = inspect(rows, origin=0.)
    assert r['all_bracketed_intervals'] == 1
    assert len(r['gaps']) == 1 and abs(r['gaps'][0]['gap_ms']-200.) < 1e-8
    assert r['skipped_stage_counts'] == dict(before_board_queue=1, queued_without_recorded_dispatch=1,
                                            after_board_reply=1)
