"""Attribute saved in-flight timings; no RPC, controller changes or new flights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


RESULTS=Path(__file__).resolve().parent/"results"
BATCHES=("inflight_events_sustained_20261026_v1","inflight_fast_recovery_20261026_v1")
STAGES=(
    ("sample_dispatch","sample_submitted_s","sample_helper_received_s"),
    ("sample_thread_queue","sample_helper_received_s","sample_io_started_s"),
    ("state_collision_rpc","sample_io_started_s","sample_io_finished_s"),
    ("sample_completion_delivery","sample_io_finished_s","sample_response_received_s"),
    ("frontend_wait_and_observe","sample_response_received_s","feature_start_s"),
    ("feature","feature_start_s","feature_done_s"),
    ("feature_submit_gap","feature_done_s","enqueue_s"),
    ("worker_submit_and_setup","enqueue_s","worker_enqueue_s"),
    ("worker_initial_queue","worker_enqueue_s","first_service_s"),
    ("result_to_command_submit","ready_s","io_submitted_s"),
    ("command_dispatch","io_submitted_s","command_helper_received_s"),
    ("command_thread_queue","command_helper_received_s","io_started_s"),
    ("command_rpc","io_started_s","rpc_return_s"))
read=lambda p:json.loads(p.read_text(encoding="utf-8"))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max())) if len(a) else None


def response_parts(row):
    parts={name:1000*(row[b]-row[a]) for name,a,b in STAGES}
    parts["cpu_dnn_compute"]=row["compute_ms"]
    parts["between_segments"]=1000*(row["ready_s"]-row["first_service_s"])-row["compute_ms"]
    total=1000*(row["rpc_return_s"]-row["sample_submitted_s"])
    if min(parts.values())<-.001 or abs(sum(parts.values())-total)>.001:
        raise ValueError("response stages overlap or omit time")
    return parts


def cycle_parts(previous,following):
    planned=following["planned_release_s"]
    acknowledged=previous["io_response_received_s"]
    parts={"post_command_state_rpc":1000*(previous["io_finished_s"]-previous["rpc_return_s"]),
        "post_command_completion_delivery":1000*(acknowledged-previous["io_finished_s"]),
        "calendar_or_event_wait":1000*max(0.,planned-acknowledged),
        "release_lateness_after_available":1000*(following["job_released_s"]-max(planned,acknowledged)),
        "release_to_sample_dispatch":1000*(following["sample_submitted_s"]-following["job_released_s"]),
        "next_response":1000*(following["rpc_return_s"]-following["sample_submitted_s"])}
    total=1000*(following["rpc_return_s"]-previous["rpc_return_s"])
    if min(parts.values())<-.001 or abs(sum(parts.values())-total)>.001:
        raise ValueError("control update stages overlap or omit time")
    return parts


def summarize(rows):
    parts=[response_parts(r) for r in rows]
    return {key:stats([p[key] for p in parts]) for key in parts[0]} if parts else {}


def intervals_overlap(segments,start,end):
    return sum(max(0.,min(s["end_s"],end)-max(s["start_s"],start)) for s in segments)


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    summaries,detail,stage_csv,cycle_csv,event_csv=[],[],[],[],[]
    inputs={}
    for batch in BATCHES:
        root=RESULTS/batch
        for directory in sorted(p.parent for p in root.glob("*/report.json")):
            report=read(directory/"report.json")
            requests=read(directory/"requests.json")
            sensor=read(directory/"sensor_samples.json")
            decisions=read(directory/"sensor_decisions.json")
            segments=read(directory/"segments.json")
            control=read(directory/"control_traces.json")
            outcomes=read(directory/"event_outcomes.json")
            analyses=read(directory/"analyses.json")
            analysis_by_id={a["sample_id"]:a for a in analyses}
            initial=read(directory/"initial_state.json")
            for name in ("report","requests","sensor_samples","sensor_decisions","segments","control_traces","event_outcomes","initial_state","analyses"):
                inputs[str((directory/(name+".json")).relative_to(RESULTS))]=digest(directory/(name+".json"))
            name=directory.name
            origin,duration=report["wall_origin_s"],report["duration_s"]
            applied=[r for r in requests if r.get("status")=="applied"]
            pulse=outcomes.get("injection",{}).get("pulse")
            vehicle=pulse["vehicle"] if pulse else ("Drone7" if len(initial)>1 else "Drone1")
            onset=outcomes.get("injection",{}).get("rpc_return_s",origin)
            critical=[r for r in applied if r["vehicle"]==vehicle]
            window=[r for r in critical if onset<=r["sample_submitted_s"] and r["rpc_return_s"]<=onset+.4]
            full=summarize(applied)
            event=summarize(window) if pulse else {}
            for scope,values in (("whole_flight",full),("critical_first400ms",event)):
                for key,distribution in values.items():
                    stage_csv.append(dict(batch=batch,trial=name,scope=scope,stage=key,**distribution))
            cycles=[]
            for previous,following in zip(critical,critical[1:]):
                if pulse and onset<=following["rpc_return_s"]<=onset+.4:
                    parts=cycle_parts(previous,following)
                    cycles.append(dict(previous=previous["sample_id"],following=following["sample_id"],
                        requested_period_ms=1000*following["requested_period_s"],
                        update_gap_ms=1000*(following["rpc_return_s"]-previous["rpc_return_s"]),parts_ms=parts))
                    cycle_csv.append(dict(batch=batch,trial=name,**{k:v for k,v in cycles[-1].items() if k!="parts_ms"},**parts))
            live_sensors=[s for s in sensor if 0.<=(s["sim_ns"]-initial[s["vehicle"]]["sim_ns"])/1e9<duration]
            live_applied=[r for r in applied if origin<=r["rpc_return_s"]<origin+duration]
            live_inputs=[r for r in requests if origin<=r["sample_submitted_s"]<origin+duration]
            busy=intervals_overlap(segments,origin,origin+duration)/duration
            summary=dict(batch=batch,trial=name,active_loops=len(initial),success=report["success"],collisions=report["collisions"],
                response_mean_ms=report["response_ms"]["mean"],cpu_dnn_mean_ms=full["cpu_dnn_compute"]["mean"],
                non_dnn_mean_ms=report["response_ms"]["mean"]-full["cpu_dnn_compute"]["mean"],
                worker_segment_busy_fraction=busy,detector_recorded_wall_ms=sum(d["detection_ms"] for d in decisions),
                detector_recorded_wall_fraction=sum(d["detection_ms"] for d in decisions)/1000/duration,
                selector_recorded_wall_fraction=report["selection_ms"]["mean"]*report["selection_ms"]["count"]/1000/duration,
                response_reserve_ms=float(np.mean([1000*analysis_by_id[r["sample_id"]]["response_reserve_s"] for r in applied])),
                responses_exceeding_reserve=sum(r["rpc_return_s"]-r["sample_submitted_s"]>analysis_by_id[r["sample_id"]]["response_reserve_s"] for r in applied),
                expired_before_release=sum(r["job_released_s"]>r["deadline_s"] for r in applied),applied_jobs=len(applied),
                input_rpc_count=2*len(live_inputs),command_and_feedback_rpc_count=4*len(live_applied),sensor_rpc_count=2*len(live_sensors),
                inferred_rpc_rate_hz=(2*len(live_inputs)+4*len(live_applied)+2*len(live_sensors))/duration)
            summaries.append(summary)
            first=None
            if pulse:
                fresh=next(r for r in critical if r["sample_submitted_s"]>=onset)
                index=critical.index(fresh)
                previous=critical[index-1] if index else None
                edge=next((d for d in decisions if d["vehicle"]==vehicle and d["capture_s"]>=onset and d["edge"]),None)
                keys=("job_released_s","sample_submitted_s","sample_s","sample_io_finished_s","sample_response_received_s","feature_start_s",
                    "feature_done_s","worker_enqueue_s","first_service_s","ready_s","io_submitted_s","io_started_s","rpc_return_s")
                times={key:1000*(fresh[key]-onset) for key in keys}
                row=dict(trial=name,batch=batch,vehicle=vehicle,sample_id=fresh["sample_id"],
                    **{key:1000*(fresh[key]-onset) for key in keys},
                    **response_parts(fresh))
                event_csv.append(row)
                raw=next((x for x in control[vehicle] if x["sample_s"]==fresh["sample_s"]),None)
                first=dict(sample_id=fresh["sample_id"],times_since_pulse_ms=times,response_parts_ms=response_parts(fresh),
                    planned_response_reserve_ms=1000*analysis_by_id[fresh["sample_id"]]["response_reserve_s"],
                    deadline_at_release_margin_ms=1000*(fresh["deadline_s"]-fresh["job_released_s"]),
                    input_sim_ns=raw["sim_ns"] if raw else None,
                    sensor_input_ms=1000*(fresh.get("sensor_capture_s",onset)-onset) if "sensor_capture_s" in fresh else None,
                    sensor_receipt_ms=1000*(fresh["sensor_received_s"]-onset) if "sensor_received_s" in fresh else None,
                    event_installed_ms=1000*(fresh["event_installed_s"]-onset) if "event_installed_s" in fresh else None,
                    previous_job={key:1000*(previous[key]-onset) for key in ("sample_s","rpc_return_s","io_finished_s","io_response_received_s")} if previous else None,
                    edge=edge,previous_stale_commands_after_pulse=sum(r["sample_submitted_s"]<onset<=r["rpc_return_s"] for r in critical))
            detail.append(dict(batch=batch,trial=name,whole_response_ms=full,critical_first400ms_response_ms=event,
                critical_first400ms_counts=dict(fresh_jobs=len(window),
                    above_reserve=sum(r["rpc_return_s"]-r["sample_submitted_s"]>analysis_by_id[r["sample_id"]]["response_reserve_s"] for r in window),
                    expired_before_release=sum(r["job_released_s"]>r["deadline_s"] for r in window),
                    late_returns=sum(r["rpc_return_s"]>r["deadline_s"] for r in window)),
                critical_cycles=cycles,first_fresh=first))
    result=dict(scope="saved17trials; accounting, not a new causal intervention or FPGA timing measurement",
        pure_cpu_compute="sum of complete FC/ReLU segment durations; inter-segment gaps separate",
        delay_limit="timestamps do not separate helper completion polling, IPC delivery and parent scheduling inside completion_delivery",
        rpc_count_semantics="code-derived calls in task window, excludes initialization and injector calls; not a packet capture",
        summaries=summaries,details=detail,input_sha256=inputs)
    (output/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for name,rows in (("trials",summaries),("response_stages",stage_csv),("critical_cycles",cycle_csv),("first_fresh",event_csv)):
        with (output/(name+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_name={d["trial"]:d["critical_first400ms_response_ms"] for d in detail}
    seven=by_name["d1_sustained_a1.95_r1_adaptive_twosided"]
    low=[by_name[n] for n in ("source_d1_T20_r1","source_d1_T20_r2")]
    groups={"input_dispatch_and_queries":("sample_dispatch","sample_thread_queue","state_collision_rpc"),
        "sample_completion_and_frontend":("sample_completion_delivery","frontend_wait_and_observe"),
        "feature":("feature","feature_submit_gap"),
        "inference_submission_and_initial_queue":("worker_submit_and_setup","worker_initial_queue"),
        "cpu_dnn_compute":("cpu_dnn_compute",),"between_segments":("between_segments",),
        "output_dispatch":("result_to_command_submit","command_dispatch","command_thread_queue"),
        "command_rpc":("command_rpc",)}
    comparison=[]
    for group,keys in groups.items():
        a=sum(seven[k]["mean"] for k in keys)
        b=sum(sum(s[k]["mean"]*s[k]["n"] for s in low)/sum(s[k]["n"] for s in low) for k in keys)
        comparison.append(dict(stage=group,seven_dynamic_ms=a,one_T20_success_pooled_ms=b,excess_ms=a-b))
    with (output/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    shutil.copy2(__file__,output/"analysis_source.py")
    for s in summaries:
        print(s["trial"],"R",round(s["response_mean_ms"],2),"DNN",round(s["cpu_dnn_mean_ms"],2),
            "busy",round(s["worker_segment_busy_fraction"],2),"RPC/s",round(s["inferred_rpc_rate_hz"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)


if __name__=="__main__":
    main()
