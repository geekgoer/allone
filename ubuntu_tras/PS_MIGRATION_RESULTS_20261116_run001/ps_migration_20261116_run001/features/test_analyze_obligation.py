#!/usr/bin/env python3
"""Synthetic lifecycle logs only: no CPU DNN, model, board or AirSim execution."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import analyze_obligation as a


def identity():
    return dict(stage="obligation_live",origin_us=1000000,duration_s=.15,period_ms=50,policy="adaptive",enable_skip=False,
        live_config=dict(names=["A"],periods_ms=[20,30,50,80],candidate_ids=[5,11]))


def request(cycle,release,period,version,model=11):
    event=dict(vehicle="A",cycle=cycle,release_ms=release,published_ms=release+.1,period_ms=period,
        deadline_ms=release+period,model_id=11,status="sampling",phase="sampling",control_version=version)
    finish=dict(event,phase="finished",status="delivered",board_done_ms=release+2,board_success=True,
        command_begin_ms=release+2.1,command_return_ms=release+3,command_pending=False,
        actuation_deadline_ms=release+period)
    return dict(vehicle="A",cycle=cycle,period_ms=period,planned_release_us=int(release*1000),
        period_deadline_us=int((release+period)*1000),model_id=model,status="delivered",calendar_release=event,
        inference_dispatched=True,calendar_dispatch=dict(event,dispatch=True,configuration_id=model),
        dpu_submit_us=int((release+1)*1000),dpu_result_readable_us=int((release+2)*1000),
        held_version=version,command_guard_accepted=True,command_sent=True,command_may_have_been_sent=True,
        command_return_us=int((release+3)*1000),new_held_version=version+1,calendar_finished=finish)


def final_state(rows,next_ms,period=50,retired=False):
    version=sum(r.get("new_held_version") is not None for r in rows)
    return dict(active=None,pending_recovery={},planned_skip={},loops=dict(A=dict(cycle=len(rows),next_ms=next_ms,
        period_ms=period,retired=retired,control_version=version,busy=None,command_pending=False)))


def no_plans():
    return dict(installed_skips={})


class ObligationAnalysisTests(unittest.TestCase):
    def audit(self,rows,identity_override=None,final=None,plans=None):
        errors=[];i=identity_override or identity()
        r=a.audit_lifecycle(rows,i,final or final_state(rows,1150),plans or no_plans(),errors)
        return r,errors

    def test_fixed_admission_counts_actual_cycles(self):
        rows=[request(k,1000+k*50,50,k) for k in range(3)]
        r,e=self.audit(rows)
        self.assertEqual(e,[])
        self.assertTrue(r["vehicles"]["A"]["nonretired_full_window_covered"])
        self.assertEqual(r["actual_inference_model_histogram"],{"11":3})
        self.assertEqual(r["vehicles"]["A"]["final_known_held_version"],3)

    def test_retimed_period_never_assumes_three_releases(self):
        rows=[request(0,1000,50,0),request(1,1080,80,1,5)]
        r,e=self.audit(rows,final=final_state(rows,1160,80))
        self.assertEqual(e,[])
        self.assertTrue(r["vehicles"]["A"]["nonretired_full_window_covered"])
        self.assertEqual(r["released_period_histogram_ms"],{"50":1,"80":1})
        self.assertEqual(r["actual_inference_model_histogram"],{"11":1,"5":1})

    def test_drain_and_retirement_are_not_full_window_pass(self):
        rows=[request(0,1000,50,0)]
        r,e=self.audit(rows,final=final_state(rows,1050,retired=True))
        self.assertEqual(e,[])
        self.assertTrue(r["vehicles"]["A"]["retired_coverage_exemption"])
        self.assertFalse(r["vehicles"]["A"]["nonretired_full_window_covered"])

    def test_missing_cycle_cutoff_and_wrong_model_are_detected(self):
        rows=[request(0,1000,50,0),request(2,1150,50,1,9)]
        _,e=self.audit(rows,final=final_state(rows,1200))
        text=" ".join(x["error"] for x in e)
        self.assertIn("contiguous",text)
        self.assertIn("cutoff",text)
        self.assertIn("disallowed model",text)

    def test_held_version_and_command_guard_audited(self):
        rows=[request(0,1000,50,0)]
        rows[0]["new_held_version"]=3;rows[0]["command_guard_accepted"]=False
        _,e=self.audit(rows,final=final_state(rows,1050,retired=True))
        text=" ".join(x["error"] for x in e)
        self.assertIn("increment",text)
        self.assertIn("guarded",text)

    def test_skip_has_no_inference_and_existing_successor_recovers(self):
        i=identity();i["enable_skip"]=True
        r0=request(0,1000,50,0)
        skip=request(1,1050,50,1)
        for key in ["inference_dispatched","calendar_dispatch","dpu_submit_us","dpu_result_readable_us","held_version","command_guard_accepted","command_sent","command_may_have_been_sent","command_return_us","new_held_version"]:
            skip.pop(key,None)
        skip["status"]="planned_skip";skip["calendar_release"]["status"]="planned_skip"
        skip["calendar_finished"]=dict(skip["calendar_release"],phase="finished",status="planned_skip")
        successor=request(2,1100,50,1);successor["calendar_release"]["recovery_due_ms"]=1140
        successor["calendar_finished"]["recovery_on_time"]=True
        plans=dict(installed_skips={"A:1":dict(job="A:1",vehicle="A",cycle=1,successor="A:2",recovery_cycle=2,
            recovery_due_ms=1140,commit_ms=1020)})
        rows=[r0,skip,successor]
        r,e=self.audit(rows,i,final_state(rows,1150),plans)
        self.assertEqual(e,[])
        self.assertEqual(r["observed_planned_skips"],1)
        self.assertEqual(r["actual_inference_count"],2)
        self.assertEqual(r["recoveries_on_time"],1)
        self.assertEqual(r["skip_recovery"][0]["outcome"],"RECOVERED_ON_TIME")
        skip["dpu_submit_us"]=1051000
        _,e=self.audit(rows,i,final_state(rows,1150),plans)
        self.assertIn("skip executed inference", " ".join(x["error"] for x in e))

    def test_outstanding_recovery_not_called_complete(self):
        rows=[request(0,1000,50,0)]
        f=final_state(rows,1050,retired=True);f["pending_recovery"]={"A":dict(cycle=2,deadline_ms=1140)}
        p=dict(installed_skips={"A:1":dict(job="A:1",vehicle="A",cycle=1,successor="A:2",recovery_cycle=2,recovery_due_ms=1140,commit_ms=1020)})
        r,e=self.audit(rows,final=f,plans=p)
        self.assertEqual(e,[])
        self.assertFalse(r["no_outstanding_recovery_or_skip"])
        self.assertEqual(r["skip_recovery"][0]["outcome"],"OUTSTANDING_AT_WINDOW_END")

    def test_planner_events_not_request_deduplicated_and_bounds_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"plans.jsonl"
            record=dict(event="obligation_plan",decision=dict(action="no_feasible_checked_plan",checks=8,candidates=9),
                install=dict(installed=False,status="no_feasible_checked_plan"),wall_ms=.3,cpu_ms=.2)
            p.write_text(json.dumps(record)+"\n"+json.dumps(record)+"\n")
            errors=[];rows=a.read_plans(p,errors);r=a.audit_plans(rows,identity(),errors)
            self.assertEqual(errors,[]);self.assertEqual(r["events"],2)
            rows[0]["decision"]["checks"]=9
            a.audit_plans(rows,identity(),errors)
            self.assertTrue(errors)

    def test_installed_skip_identity_and_release_updates(self):
        record=dict(event="obligation_plan",decision=dict(accepted=True,action="skip_recover",checks=3,candidates=3,commit_by=1,
            skipped="A:1",permission=dict(job="A:1",successor="A:2",hold_until=120,eligible=True,evidence="synthetic qualified observed hold"),choice={"A:2":5}),
            install=dict(installed=True,elapsed_ms=.1,commit_ms=1020.1,updates=[dict(vehicle="A",cycle=1,old_release_ms=1050,next_release_ms=1080,old_period_ms=50,period_ms=80)]),
            snapshot=dict(snapshot_ms=1020,calendar=dict(jobs=[dict(name="A:2",current_mode=11)])),wall_ms=.1,cpu_ms=.1)
        i=identity();i["enable_skip"]=True;errors=[]
        r=a.audit_plans([record],i,errors)
        self.assertEqual(errors,[])
        self.assertEqual(len(r["installed_release_updates"]),1)
        self.assertEqual(len(r["installed_model_edits"]),1)
        self.assertEqual(r["installed_skips"]["A:1"]["recovery_due_ms"],1140)

    def test_full_wrapper_accepts_two_retimed_cycles_not_fixed_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=[request(0,1000,50,0),request(1,1080,80,1)]
            (p/"execution_identity.json").write_text(json.dumps(identity()))
            (p/"calendar_final.json").write_text(json.dumps(final_state(rows,1160,80)))
            (p/"events.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            (p/"summary.json").write_text(json.dumps(dict(early_stop="",journal_failed=False)))
            r,_,_=a.analyze_run(p)
            self.assertTrue(r["audit_passed"])
            self.assertTrue(r["calendar_timing_acceptance"])
            self.assertNotIn("expected_planned_slots",r["vehicles"]["A"])
            self.assertTrue(r["vehicles"]["A"]["release_records_complete"])
            self.assertFalse(r["complete_flight_observed"])


if __name__=="__main__":unittest.main(verbosity=2)
