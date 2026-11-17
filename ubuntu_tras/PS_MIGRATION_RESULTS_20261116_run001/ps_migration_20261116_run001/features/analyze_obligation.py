#!/usr/bin/env python3
"""Audit actual Route-B obligation-calendar logs, without running any model.

Unlike the fixed-period analyzer, release coverage comes from the recorded
calendar lifecycle and final next-release state, not ceil(duration/period).
Python 3.6+ standard library only; raw directories are never modified.
"""
from __future__ import print_function

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import analyze_flight as base


def fail(errors, message, **context):
    context["error"] = message
    errors.append(context)


def close(a,b,tolerance=.002):
    return base.finite_number(a) and base.finite_number(b) and abs(a-b)<=tolerance


def read_plans(path, errors):
    """Planner rows are events, NOT unique vehicle/cycle request rows."""
    result=[]
    if not path.is_file():
        return result
    for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            item=json.loads(line)
            if not isinstance(item,dict):
                raise ValueError("planner event is not an object")
            result.append(item)
        except ValueError as error:
            fail(errors,str(error),file=str(path),line=number)
    return result


def audit_plans(plans,identity,errors):
    actions=collections.Counter();installed=collections.Counter();rejected=collections.Counter()
    wall=[];cpu=[];checks=[];candidates=[];skips={};updates=[];mode_edits=[]
    for number,item in enumerate(plans):
        if item.get("event")!="obligation_plan":
            fail(errors,"non-plan event in planner log",event=item.get("event"),index=number)
            continue
        decision=item.get("decision",{});commit=item.get("install",{});snapshot=item.get("snapshot")
        action=str(decision.get("action","missing"));actions[action]+=1
        if base.finite_number(item.get("wall_ms")):wall.append(item["wall_ms"])
        if base.finite_number(item.get("cpu_ms")):cpu.append(item["cpu_ms"])
        for key,maximum,values in [("checks",8,checks),("candidates",12,candidates)]:
            value=decision.get(key)
            if value is not None:
                if not isinstance(value,int) or isinstance(value,bool) or value<0 or value>maximum:
                    fail(errors,"planner search bound exceeded/invalid",index=number,field=key,value=value,maximum=maximum)
                else:values.append(value)
        if commit.get("installed") is not True:
            rejected[str(commit.get("status","missing"))]+=1
            continue
        installed[action]+=1
        if identity.get("policy")=="fixed":
            fail(errors,"fixed baseline installed an adaptive plan",index=number)
        if decision.get("accepted") is not True or not isinstance(snapshot,dict):
            fail(errors,"installed plan lacks an accepted decision and snapshot",index=number)
            continue
        commit_ms=commit.get("commit_ms")
        if not close(commit.get("elapsed_ms"),commit_ms-snapshot.get("snapshot_ms",0.) if base.finite_number(commit_ms) else None):
            fail(errors,"installed elapsed/snapshot clock mismatch",index=number)
        if base.finite_number(commit.get("elapsed_ms")) and base.finite_number(decision.get("commit_by")):
            if commit["elapsed_ms"]>decision["commit_by"]+1e-6:
                fail(errors,"plan installed beyond checked commit budget",index=number)
        previous={j.get("name"):j.get("current_mode") for j in snapshot.get("calendar",{}).get("jobs",[])}
        for job,mode in decision.get("choice",{}).items():
            if previous.get(job)!=mode:
                mode_edits.append(dict(plan_index=number,job=job,old_model_id=previous.get(job),new_model_id=mode,commit_ms=commit_ms))
        for update in commit.get("updates",[]):
            record=dict(update);record["plan_index"]=number;record["commit_ms"]=commit_ms;updates.append(record)
        target=decision.get("skipped")
        if target is not None:
            if identity.get("enable_skip") is not True:
                fail(errors,"skip installed without explicit opt-in",index=number,job=target)
            permission=decision.get("permission") or {}
            successor=permission.get("successor")
            if permission.get("eligible") is not True or not permission.get("evidence"):
                fail(errors,"installed skip lacks eligible qualified permission evidence",index=number,job=target)
            if permission.get("job")!=target or not isinstance(target,str) or not isinstance(successor,str):
                fail(errors,"skip permission identity incomplete",index=number)
                continue
            if target in skips:
                fail(errors,"same skip obligation installed twice",job=target)
            try:
                vehicle,cycle=target.rsplit(":",1);other,next_cycle=successor.rsplit(":",1)
                if vehicle!=other or int(next_cycle)!=int(cycle)+1:
                    raise ValueError("skip successor must be the immediate same-vehicle cycle")
            except ValueError as error:
                fail(errors,str(error),job=target,successor=successor)
                continue
            deadline=snapshot.get("snapshot_ms",0.)+permission.get("hold_until",0.)
            skips[target]=dict(job=target,vehicle=vehicle,cycle=int(cycle),successor=successor,
                recovery_cycle=int(next_cycle),recovery_due_ms=deadline,commit_ms=commit_ms,
                plan_index=number,permission=permission)
    return dict(events=len(plans),actions=dict(actions),installed_actions=dict(installed),
        rejected_installations=dict(rejected),installed_count=sum(installed.values()),
        installed_release_updates=updates,installed_model_edits=mode_edits,
        wall_ms=base.stats(wall),cpu_ms=base.stats(cpu),checks=base.stats(checks),candidates=base.stats(candidates),
        installed_skips=skips)


def row_key(row):
    return "{}:{}".format(row.get("vehicle"),row.get("cycle"))


def audit_lifecycle(rows,identity,final,plans,errors):
    origin=identity.get("origin_us");duration=identity.get("duration_s")
    cutoff=origin/1000.+duration*1000. if base.finite_number(origin) and base.finite_number(duration) else None
    if cutoff is None:
        fail(errors,"missing finite origin/duration for calendar coverage")
    config=identity.get("live_config",final.get("config",{}))
    allowed=config.get("candidate_ids",[]);periods=config.get("periods_ms",[])
    loops=final.get("loops",{})
    if not isinstance(loops,dict) or not loops:
        fail(errors,"missing final calendar loops")
        loops={}
    names=set(config.get("names",[]))|set(loops)|set(r.get("vehicle") for r in rows if isinstance(r.get("vehicle"),str))
    bykey={row_key(r):r for r in rows}
    model_counts=collections.Counter();command_models=collections.Counter();release_models=collections.Counter();period_counts=collections.Counter()
    vehicles={};dpu_intervals=[];command_audit_rows=[]
    for name in sorted(names):
        selected=sorted([r for r in rows if r.get("vehicle")==name],key=lambda r:r.get("cycle",-1))
        cycles=[r.get("cycle") for r in selected]
        cycle_ok=cycles==list(range(len(selected)))
        if not cycle_ok:fail(errors,"cycles are not contiguous from zero",vehicle=name,cycles=cycles)
        last_release=None;version=0;command_returns=0;skip_count=0;released_periods=collections.Counter();models=collections.Counter()
        for row in selected:
            cycle=row.get("cycle");context=dict(vehicle=name,cycle=cycle)
            event=row.get("calendar_release")
            if not isinstance(event,dict):
                fail(errors,"calendar_release missing",**context)
                continue
            if event.get("vehicle")!=name or event.get("cycle")!=cycle:
                fail(errors,"release identity mismatch",**context)
            release=event.get("release_ms");period=event.get("period_ms")
            if not close(row.get("planned_release_us",-1)/1000.,release):
                fail(errors,"planned release timestamp disagrees with calendar",**context)
            if not close(row.get("period_ms"),period) or not base.finite_number(period) or period<=0:
                fail(errors,"released period invalid/mismatch",**context)
            else:
                period_key="{:g}".format(period);period_counts[period_key]+=1;released_periods[period_key]+=1
                if periods and not any(close(period,p,1e-6) for p in periods):
                    fail(errors,"unsupported released period",period_ms=period,**context)
            if not close(event.get("deadline_ms"),release+period if base.finite_number(release) and base.finite_number(period) else None):
                fail(errors,"release deadline does not equal release+period",**context)
            if not close(row.get("period_deadline_us",-1)/1000.,event.get("deadline_ms")):
                fail(errors,"period deadline timestamp mismatch",**context)
            if base.finite_number(release):
                if cutoff is not None and (release<origin/1000.-.002 or release>=cutoff-1e-6):
                    fail(errors,"release outside admitted measurement cutoff",release_ms=release,cutoff_ms=cutoff,**context)
                if base.finite_number(event.get("published_ms")) and event["published_ms"]+.002<release:
                    fail(errors,"calendar published before absolute release",**context)
                if base.finite_number(row.get("actual_publish_us")) and row["actual_publish_us"]/1000.+.002<release:
                    fail(errors,"native publisher timestamp precedes absolute release",**context)
                if last_release is not None:
                    if release<=last_release:fail(errors,"nonincreasing planned releases",**context)
                    if not close(release-last_release,period):
                        fail(errors,"retimed release is not previous release + its current period",**context)
                last_release=release
            release_models[str(event.get("model_id"))]+=1
            release_status=event.get("status");skip=release_status=="planned_skip"
            if skip:
                skip_count+=1
                if row.get("status")!="planned_skip":fail(errors,"planned skip changed into another terminal status",**context)
                if identity.get("enable_skip") is not True:fail(errors,"actual skip without opt-in",**context)
                if row.get("inference_dispatched") is True or any(base.finite_number(row.get(k)) and row[k]>0 for k in ["dpu_submit_us","dpu_result_readable_us","input_prep_start_us","command_attempt_start_us"]):
                    fail(errors,"planned skip executed inference or command",**context)
                if row.get("command_sent") is True or row.get("physical_output") is not None:
                    fail(errors,"planned skip contains output/actual command",**context)
            dispatch=row.get("calendar_dispatch")
            if row.get("inference_dispatched") is True:
                model=row.get("model_id");model_counts[str(model)]+=1;models[str(model)]+=1
                if allowed and model not in allowed:fail(errors,"real inference selected disallowed model",model_id=model,**context)
                if not isinstance(dispatch,dict) or dispatch.get("dispatch") is not True or dispatch.get("configuration_id")!=model:
                    fail(errors,"actual inference identity does not match dispatch",**context)
                elif dispatch.get("vehicle")!=name or dispatch.get("cycle")!=cycle:
                    fail(errors,"DPU dispatch belongs to another request",**context)
                start,end=row.get("dpu_submit_us"),row.get("dpu_result_readable_us")
                if base.finite_number(start) and base.finite_number(end) and start>0 and end>=start:
                    dpu_intervals.append((start,end,name,cycle))
            finished=row.get("calendar_finished",{})
            if not isinstance(finished,dict) or not finished:
                fail(errors,"terminal lifecycle calendar_finished missing",**context)
            elif finished.get("vehicle")!=name or finished.get("cycle")!=cycle or finished.get("phase")!="finished" or finished.get("status")!=row.get("status"):
                fail(errors,"terminal lifecycle identity/phase/status mismatch",**context)
            if base.finite_number(row.get("held_version")) and row["held_version"]!=version:
                fail(errors,"sampled held version differs from last acknowledged command",expected=version,actual=row["held_version"],**context)
            new_version=row.get("new_held_version")
            begin=finished.get("command_begin_ms") if isinstance(finished,dict) else None
            returned=finished.get("command_return_ms") if isinstance(finished,dict) else None
            if row.get("status") in ("delivered","command_return_late") and (
                    new_version is None or row.get("command_sent") is not True or row.get("inference_dispatched") is not True):
                fail(errors,"delivered status lacks actual inference and acknowledged held-version evidence",**context)
            if new_version is not None:
                command_returns+=1;command_models[str(row.get("model_id"))]+=1
                if new_version!=version+1:fail(errors,"acknowledged held version did not increment exactly once",previous=version,new=new_version,**context)
                version=new_version
                if row.get("command_guard_accepted") is not True or row.get("command_sent") is not True:
                    fail(errors,"held version advanced without guarded actual command",**context)
                if not close(returned,row.get("command_return_us",-1)/1000.) or not base.finite_number(begin) or begin>returned:
                    fail(errors,"command begin/return lifecycle mismatch",**context)
                if base.finite_number(begin) and base.finite_number(row.get("command_attempt_start_us")) and begin>row["command_attempt_start_us"]/1000.+.002:
                    fail(errors,"native command preceded calendar command_begin guard",**context)
                if isinstance(finished,dict) and (finished.get("board_success") is not True or not base.finite_number(finished.get("board_done_ms")) or begin<finished["board_done_ms"]):
                    fail(errors,"command started before successful DPU completion",**context)
                if finished.get("command_pending") is not False:fail(errors,"acknowledged command remains pending in row",**context)
            elif returned is not None:
                fail(errors,"calendar acknowledged a command but raw held version absent",**context)
            if row.get("command_guard_accepted") is True or row.get("command_may_have_been_sent") is True:
                command_audit_rows.append(dict(vehicle=name,cycle=cycle,model_id=row.get("model_id"),
                    command_guard_accepted=row.get("command_guard_accepted"),command_begin_ms=begin,
                    command_return_ms=returned,held_version_before=row.get("held_version"),new_held_version=new_version,
                    delivery_known=new_version is not None,status=row.get("status")))
        loop=loops.get(name)
        coverage=False;retired=None;next_ms=None
        if not isinstance(loop,dict):
            fail(errors,"vehicle missing final calendar state",vehicle=name)
        else:
            retired=loop.get("retired") is True;next_ms=loop.get("next_ms")
            if loop.get("cycle")!=len(selected):fail(errors,"final calendar cycle differs from raw release count",vehicle=name,calendar_cycle=loop.get("cycle"),raw_count=len(selected))
            if loop.get("control_version")!=version:fail(errors,"final held version differs from raw acknowledged commands",vehicle=name,calendar_version=loop.get("control_version"),raw_version=version)
            if last_release is not None and not close(next_ms-last_release,loop.get("period_ms")):
                fail(errors,"final next release does not preserve current period",vehicle=name)
            coverage=bool(cutoff is not None and base.finite_number(next_ms) and next_ms>=cutoff-1e-6)
        statuses=collections.Counter(r.get("status","missing") for r in selected)
        all_good=bool(selected) and all(r.get("status") in ("delivered","planned_skip") and not r.get("error") for r in selected)
        drained=bool(isinstance(loop,dict) and loop.get("busy") is None and loop.get("command_pending") is False)
        vehicles[name]=dict(recorded_releases=len(selected),cycles_contiguous=cycle_ok,
            released_period_histogram_ms=dict(released_periods),actual_inference_model_histogram=dict(models),
            actual_inferences=sum(models.values()),acknowledged_commands=command_returns,final_known_held_version=version,
            planned_skips=skip_count,statuses=dict(statuses),final_loop=loop,
            next_release_ms=next_ms,cutoff_ms=cutoff,calendar_cutoff_reached=coverage,
            retired=retired,retired_coverage_exemption=retired is True and not coverage,
            lifecycle_window_covered=bool(cycle_ok and (coverage or retired is True)),
            nonretired_full_window_covered=bool(cycle_ok and coverage and retired is False),
            drained=drained,all_release_outcomes_delivered_or_planned_skip=all_good)
    dpu_intervals.sort()
    for previous,current in zip(dpu_intervals[:-1],dpu_intervals[1:]):
        if current[0]<previous[1]:fail(errors,"real DPU intervals overlap on one nonpreemptive device",previous=previous,current=current)
    if final.get("active") is not None:fail(errors,"final calendar retains an active DPU job",active=final.get("active"))
    # Planned skip and its immediate existing successor form one recorded debt.
    recovery=[]
    skips=plans["installed_skips"]
    for row in rows:
        if row.get("calendar_release",{}).get("status")=="planned_skip" and row_key(row) not in skips:
            fail(errors,"actual planned skip lacks an installed permission/plan",job=row_key(row))
    for job,record in sorted(skips.items()):
        item=dict(record);skip_row=bykey.get(job);successor=bykey.get(record["successor"])
        item["skip_release_observed"]=bool(skip_row and skip_row.get("status")=="planned_skip")
        item["successor_observed"]=successor is not None
        item["recovery_on_time"]=False
        if successor is not None:
            event=successor.get("calendar_release",{})
            due=event.get("recovery_due_ms")
            if not close(due,record["recovery_due_ms"]):fail(errors,"successor recovery deadline differs from committed hold deadline",job=record["successor"])
            finish=successor.get("calendar_finished",{})
            ack=finish.get("command_return_ms")
            deadline=min(record["recovery_due_ms"],finish.get("actuation_deadline_ms",record["recovery_due_ms"]))
            observed=base.finite_number(ack) and ack<=deadline
            item.update(successor_status=successor.get("status"),command_return_ms=ack,effective_recovery_deadline_ms=deadline,
                        recovery_on_time=observed,recorded_recovery_on_time=finish.get("recovery_on_time"))
            if (ack is not None or "recovery_on_time" in finish) and finish.get("recovery_on_time") is not observed:
                fail(errors,"recorded recovery completion differs from actual command deadline",job=record["successor"])
        item["pending_at_end"]=final.get("pending_recovery",{}).get(record["vehicle"],{}).get("cycle")==record["recovery_cycle"]
        if item["recovery_on_time"] and item["pending_at_end"]:
            fail(errors,"on-time recovery debt was not cleared",job=record["successor"])
        if item["pending_at_end"]:item["outcome"]="OUTSTANDING_AT_WINDOW_END"
        elif item["skip_release_observed"] and item["recovery_on_time"]:item["outcome"]="RECOVERED_ON_TIME"
        elif not item["skip_release_observed"]:item["outcome"]="COMMITTED_BUT_SKIP_NOT_OBSERVED"
        else:item["outcome"]="RECOVERY_NOT_PROVED"
        recovery.append(item)
    previous_skip={}
    for item in sorted(recovery,key=lambda r:r["commit_ms"]):
        previous=previous_skip.get(item["vehicle"])
        if previous is not None and (not previous["recovery_on_time"] or previous.get("command_return_ms",float("inf"))>item["commit_ms"]):
            fail(errors,"another skip committed while earlier recovery debt was outstanding",job=item["job"],previous_job=previous["job"])
        previous_skip[item["vehicle"]]=item
    return dict(vehicles=vehicles,actual_inference_model_histogram=dict(model_counts),
        acknowledged_command_model_histogram=dict(command_models),released_default_model_histogram=dict(release_models),
        released_period_histogram_ms=dict(period_counts),actual_inference_count=sum(model_counts.values()),
        real_dpu_intervals=len(dpu_intervals),command_lifecycle=command_audit_rows,
        skip_recovery=recovery,observed_planned_skips=sum(v["planned_skips"] for v in vehicles.values()),
        installed_skip_count=len(skips),recoveries_on_time=sum(r["recovery_on_time"] for r in recovery),
        pending_recovery_at_end=final.get("pending_recovery",{}),planned_skips_at_end=final.get("planned_skip",{}),
        all_observed_skips_recovered_on_time=all(r["recovery_on_time"] for r in recovery if r["skip_release_observed"]),
        no_outstanding_recovery_or_skip=not final.get("pending_recovery") and not final.get("planned_skip"),
        coverage_basis="Recorded calendar_release cycles/periods and final loop next_ms relative to cutoff; no duration/period expected-count assumption")


def analyze_run(folder):
    report,timing_rows,tracking_rows=base.analyze_run(folder)
    errors=[]
    rows=base.load_rows(Path(report["input_log"]),errors) if Path(report["input_log"]).is_file() else []
    identity=report["execution_identity"]
    if identity.get("stage")!="obligation_live":fail(errors,"not an obligation_live runtime identity")
    final=base.load_json(folder/"calendar_final.json",{})
    plan_rows=read_plans(folder/"plans.jsonl",errors)
    plan_report=audit_plans(plan_rows,identity,errors)
    life=audit_lifecycle(rows,identity,final,plan_report,errors)
    for name,vehicle in report["vehicles"].items():
        entry=life["vehicles"].get(name,{})
        vehicle.pop("expected_planned_slots",None)
        vehicle["release_records_complete"]=entry.get("nonretired_full_window_covered",False)
        vehicle["coverage_basis"]=life["coverage_basis"]
        vehicle["timing_passed"]=bool(entry.get("nonretired_full_window_covered") and entry.get("drained") and
            entry.get("all_release_outcomes_delivered_or_planned_skip") and not vehicle["counts"]["late_command_returns"])
    summary=report.get("runtime_summary") or {}
    runtime_failed=bool(report.get("failure") or summary.get("early_stop") or summary.get("interrupted") or
        summary.get("journal_failed") or summary.get("executor_failed") or summary.get("planner_failed"))
    report.update(schema="ps-obligation-observational-analysis/v1",calendar_final=final,
        calendar_lifecycle=life,planner=plan_report,calendar_audit_errors=errors,
        audit_passed=report["audit_passed"] and not errors,runtime_failed=runtime_failed)
    report["complete_flight_observed"]=bool(report["complete_flight_observed"] and not runtime_failed and not errors)
    report["calendar_timing_acceptance"]=bool(report["vehicles"] and report["audit_passed"] and not runtime_failed and
        all(v["timing_passed"] for v in report["vehicles"].values()) and life["no_outstanding_recovery_or_skip"] and
        life["all_observed_skips_recovered_on_time"])
    initial_model=identity.get("model_id",identity.get("live_config",{}).get("model_id"))
    initial_period=identity.get("period_ms")
    report["installed_strategy_edits"]=bool(plan_report["installed_model_edits"] or plan_report["installed_release_updates"] or plan_report["installed_skips"])
    report["strategy_effect_observed"]=bool(life["observed_planned_skips"] or
        any(str(initial_model)!=key for key in life["actual_inference_model_histogram"] if initial_model is not None) or
        any(not close(float(key),initial_period) for key in life["released_period_histogram_ms"] if base.finite_number(initial_period)))
    report["skip_recovery_experiment_validated"]=bool(identity.get("enable_skip") is True and life["observed_planned_skips"]>0 and
        report["calendar_timing_acceptance"] and life["all_observed_skips_recovered_on_time"])
    for p in [folder/"calendar_final.json",folder/"plans.jsonl",folder/"config_source.json"]:
        if p.is_file():report["sha256"][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return report,timing_rows,tracking_rows


def markdown(report):
    lines=["# Route-B obligation-calendar PS analysis","",
        "Calendar releases are audited from their actual lifecycle, not an assumed fixed number of slots. All timing is empirical; no WCET or pure hardware timing claim.",""]
    if report.get("label"):lines.extend(["Classification: `{}`.".format(report["label"]),""])
    for run in report["runs"]:
        life=run["calendar_lifecycle"];planner=run["planner"];identity=run["execution_identity"]
        lines.extend(["## "+run["data"],"",
            "Policy `{}`; skip opt-in `{}`; audit `{}`; runtime failure `{}`; calendar/timing acceptance `{}`; observed complete flight `{}`.".format(
                identity.get("policy"),identity.get("enable_skip"),run["audit_passed"],run["runtime_failed"],run["calendar_timing_acceptance"],run["complete_flight_observed"]),"",
            "Statuses: `{}`.".format(json.dumps(run["counts"]["status"],sort_keys=True)),"",
            "Actual DPU model histogram: `{}`. Released period histogram (ms): `{}`.".format(json.dumps(life["actual_inference_model_histogram"],sort_keys=True),json.dumps(life["released_period_histogram_ms"],sort_keys=True)),"",
            "Installed actions: `{}`. Model edits `{}`; release edits `{}`; observed skips `{}`; on-time recoveries `{}`; outstanding debts `{}`.".format(
                json.dumps(planner["installed_actions"],sort_keys=True),len(planner["installed_model_edits"]),len(planner["installed_release_updates"]),life["observed_planned_skips"],life["recoveries_on_time"],json.dumps(life["pending_recovery_at_end"],sort_keys=True)),""])
        for name,v in sorted(life["vehicles"].items()):
            lines.extend(["- {}: {} actual release records; cutoff reached `{}`; retired `{}`; drained `{}`; acknowledged commands {}; held version {}.".format(
                name,v["recorded_releases"],v["calendar_cutoff_reached"],v["retired"],v["drained"],v["acknowledged_commands"],v["final_known_held_version"]),""])
        lines.extend(["| PS phase (ms) | Samples | P50 | P95 | P99 | Maximum observed |","| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name,_,_ in base.PHASES:
            s=run["timing_ms"][name]
            if s["count"]:lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(name,s["count"],s["p50"],s["p95"],s["p99"],s["max"]))
        lines.append("")
        if identity.get("policy")=="adaptive" and not run["strategy_effect_observed"]:
            lines.extend(["No actual mode/period/skip strategy effect was observed; an adaptive policy label alone does not demonstrate an adaptation advantage.",""])
        if run.get("failure"):lines.extend(["Preserved runtime failure: `{}`.".format(json.dumps(run["failure"])),""])
    lines.extend(["## Limits","", "Planned skips are deliberate non-inference releases, not dropped packets. Successful debt recovery requires an actual acknowledged successor command before the intersected deadline, not merely a planned recovery. Pending commitments at the measurement cutoff remain outstanding. Flight, transport, timing and strategy effectiveness are separate results.",""])
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,nargs="+",type=Path)
    parser.add_argument("--output",required=True,type=Path,help="NEW analysis directory")
    parser.add_argument("--label",default=None)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=False,exist_ok=False)
    runs=[];timings=[];tracks=[];commands=[];debts=[];updates=[];models=[];summaries=[];vehicle_summaries=[]
    for folder in args.data:
        report,timing,track=analyze_run(folder);runs.append(report)
        summaries.append(dict(run=str(folder),policy=report["execution_identity"].get("policy"),
            skip_enabled=report["execution_identity"].get("enable_skip"),audit_passed=report["audit_passed"],
            runtime_failed=report["runtime_failed"],calendar_timing_acceptance=report["calendar_timing_acceptance"],
            complete_flight_observed=report["complete_flight_observed"],strategy_effect_observed=report["strategy_effect_observed"],
            skip_recovery_experiment_validated=report["skip_recovery_experiment_validated"],
            statuses=report["counts"]["status"],actual_model_histogram=report["calendar_lifecycle"]["actual_inference_model_histogram"],
            released_period_histogram_ms=report["calendar_lifecycle"]["released_period_histogram_ms"],
            installed_actions=report["planner"]["installed_actions"],
            release_command_p50_ms=report["timing_ms"]["release_to_command_proxy"]["p50"],
            release_command_p99_ms=report["timing_ms"]["release_to_command_proxy"]["p99"],
            release_command_max_ms=report["timing_ms"]["release_to_command_proxy"]["max"],
            dpu_call_p99_ms=report["timing_ms"]["dpu_call"]["p99"],planner_wall_max_ms=report["planner"]["wall_ms"]["max"]))
        for name,v in report["vehicles"].items():
            t=v["tracking"];g=v["gate_crossings"];life=report["calendar_lifecycle"]["vehicles"].get(name,{})
            vehicle_summaries.append(dict(run=str(folder),vehicle=name,policy=report["execution_identity"].get("policy"),
                recorded_releases=life.get("recorded_releases"),nonretired_full_window_covered=life.get("nonretired_full_window_covered"),
                held_version=life.get("final_known_held_version"),sample_rmse_m=t.get("observed_sample_rmse_m"),
                full_window_observed_rmse_m=t.get("full_window_tracking_rmse_m"),
                max_position_error_m=t.get("observed_position_error_m",{}).get("max"),sim_wall_ratio=t.get("sim_wall_ratio"),
                gate_count=g.get("expected_gates"),all_body_crossings_observed=g.get("all_body_crossings_observed"),
                final_snapshot_new_collisions=v["final_snapshot_new_collisions"]))
        for collection,records in [(timings,timing),(tracks,track),(commands,report["calendar_lifecycle"]["command_lifecycle"]),
                (debts,report["calendar_lifecycle"]["skip_recovery"]),(updates,report["planner"]["installed_release_updates"]),
                (models,report["planner"]["installed_model_edits"])]:
            for row in records:
                item=dict(row);item["run"]=str(folder);collection.append(item)
    result=dict(schema="ps-obligation-analysis-collection/v1",label=args.label,runs=runs,
        audit_passed=all(r["audit_passed"] for r in runs),calendar_timing_acceptances=sum(r["calendar_timing_acceptance"] for r in runs),
        complete_flights_observed=sum(r["complete_flight_observed"] for r in runs))
    (args.output/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    (args.output/"REPORT.md").write_text(markdown(result),encoding="utf-8")
    for name,rows in [("timing_per_request",timings),("tracking_samples",tracks),("command_lifecycle",commands),
            ("skip_recovery",debts),("installed_release_updates",updates),("installed_model_edits",models)]:
        base.write_csv(args.output/(name+".csv"),rows)
    base.write_csv(args.output/"run_summary.csv",summaries)
    base.write_csv(args.output/"vehicle_summary.csv",vehicle_summaries)
    print("PS_OBLIGATION_ANALYSIS runs={} audit={} calendar_timing_acceptances={} observed_complete_flights={} output={}".format(
        len(runs),"PASS" if result["audit_passed"] else "FAIL",result["calendar_timing_acceptances"],result["complete_flights_observed"],args.output))
    return 0 if result["audit_passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
