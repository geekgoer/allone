"""Freeze empirical CPU pipeline costs from an explicitly selected development run."""

import argparse
import json
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json, sha256
from .framework_runtime_flight import DEFAULT_COSTS
from .framework_runtime import TIMING_CONTRACT


FIELDS = {"input_s":("feature_done_s","frontend_start_s"),
    "output_s":("output_finished_s","handled_s"),
    "sample_io_s":("sample_io_finished_s","sample_submitted_s"),
    "command_s":("rpc_return_s","io_submitted_s"),
    "actuation_s":("rpc_return_s","io_started_s"),
    "io_s":("io_finished_s","io_submitted_s"),
    "ingress_s":("worker_enqueue_s","enqueue_s"),
    "egress_s":("received_s","ready_s"),
    "handoff_s":("io_submitted_s","io_handoff_s")}
VISIBLE_FIELDS = {**FIELDS,"sample_io_s":("sample_response_received_s","sample_submitted_s"),
    "command_s":("command_response_received_s","io_submitted_s"),
    "io_s":("io_response_received_s","io_submitted_s"),
    "ingress_s":("worker_accept_received_s","enqueue_s")}


def stage_measurements(rows, decisions, fields):
    # Parent receipt timestamps gathered by close() after an abort include
    # cleanup waiting, not normal pipeline visibility. Never train costs on it.
    active_keys = ("sample_submitted_s","frontend_start_s","feature_done_s","enqueue_s",
        "received_s","io_handoff_s","io_submitted_s","handled_s","output_finished_s")
    active = [r[k] for r in rows for k in active_keys if k in r]
    active += [r["finished_s"] for r in decisions if not r.get("outside_physics") and "finished_s" in r]
    if not active:
        raise ValueError("no active parent pipeline observations")
    cutoff = max(active)
    values = {name:[r[end]-r[start] for r in rows if end in r and start in r and r[end]<=cutoff
        and (name not in ("command_s","io_s") or "rpc_return_s" in r)]
        for name,(end,start) in fields.items()}
    return values,cutoff


def decomposed_measurements(rows, decisions):
    """Remove only known frontend blocking AFTER a helper operation completes.

    Serialized frontend work remains charged by the pipeline model. All helper
    service, unexplained IPC/OS wait and notification delay outside that work
    remain in these empirical estimates. This is not a WCET decomposition.
    """
    visible,cutoff = stage_measurements(rows,decisions,VISIBLE_FIELDS)
    intervals = [(r[a],r[b]) for r in rows for a,b in
        (("frontend_start_s","feature_done_s"),("handled_s","output_finished_s"))
        if a in r and b in r and r[b]<=cutoff]
    intervals += [(d["started_s"],d["finished_s"]) for d in decisions
        if not d.get("outside_physics") and "started_s" in d and d["finished_s"]<=cutoff]
    merged = []
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:
            merged[-1][1] = max(b,merged[-1][1])
        else:
            merged.append([a,b])
    measured = {k:list(v) for k,v in visible.items()}
    for name in ("sample_io_s","command_s","io_s","ingress_s","egress_s"):
        end,start = VISIBLE_FIELDS[name]
        if name=="command_s":
            # The job deadline constrains actuation (the RPC-return proxy),
            # not the parent's later acknowledgement. Feedback visibility is
            # still charged by io_s; gross acknowledgement tails remain saved.
            end = "rpc_return_s"
        helper_end = "ready_s" if name=="egress_s" else FIELDS[name][0]
        measured[name] = []
        for r in rows:
            if not all(k in r for k in (start,helper_end,end)) or r[end]>cutoff:
                continue
            if name in ("command_s","io_s") and "rpc_return_s" not in r:
                continue
            overlap = sum(max(0.,min(r[end],b)-max(r[helper_end],a)) for a,b in merged)
            measured[name].append(r[end]-r[start]-overlap)
    return measured,visible,cutoff


def freeze(source, output, statistic="maximum", decompose_frontend_wait=False):
    setup = json.loads((source/"setup.json").read_text())
    if setup.get("version") not in ("framework_runtime_v6_affinity","framework_runtime_v13_editor_condition","framework_runtime_v14_stage_visibility","framework_runtime_v17_release_priority","framework_runtime_v18_ack_progress","framework_runtime_v19_compact_io","framework_runtime_v20_deferred_admission","framework_runtime_v21_stall_diagnostics","framework_runtime_v22_event_executor","framework_runtime_v23_precise_release","framework_runtime_v25_direct_collision","framework_runtime_v26_fine_static","framework_runtime_v27_stable_static","framework_runtime_v29_context_grid","framework_runtime_v37_current_costs"):
        raise ValueError("use an explicit matching CPU development run, not an older hardware/protocol profile")
    if setup.get("timing_contract") != TIMING_CONTRACT:
        raise ValueError("cost measurements predate the current state/collision snapshot contract")
    fields = VISIBLE_FIELDS
    values = {name:[] for name in fields}
    visible_values = {name:[] for name in fields}
    values["planner_reserve_s"] = []
    hashes = {"setup.json":sha256(source/"setup.json")}
    cutoffs = {}
    for policy in ("static","adaptive"):
        directory = source/policy
        report = json.loads((directory/"report.json").read_text())
        if "communication" not in report["cpu_placement"] or report["error"]:
            raise ValueError("calibration requires the independent communication process and no infrastructure error")
        for filename in ("report.json","requests.json","decisions.json"):
            hashes[f"{policy}/{filename}"] = sha256(directory/filename)
        rows = json.loads((directory/"requests.json").read_text())
        decisions = json.loads((directory/"decisions.json").read_text())
        if decompose_frontend_wait:
            measured,visible,cutoffs[policy] = decomposed_measurements(rows,decisions)
        else:
            measured,cutoffs[policy] = stage_measurements(rows,decisions,fields)
            visible = measured
        for name,samples in measured.items():
            values[name] += samples
            visible_values[name] += visible[name]
        values["planner_reserve_s"] += [r["overhead_ms"]/1000 for r in decisions
            if not r.get("outside_physics")]
    if any(not v or min(v)<0 or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("all stages need finite nonnegative development measurements")
    costs = dict(DEFAULT_COSTS)
    # A declared heuristic, not tuned against the next acceptance run and NOT WCET.
    for name,samples in values.items():
        estimate = max(samples) if statistic == "maximum" else float(np.percentile(samples,95))
        costs[name] = max(.001,1.25*estimate+.0005)
    costs["io_s"] = max(costs["io_s"],costs["command_s"])
    result = {"contract":"framework_cpu_empirical_costs_v1","vehicles":setup["vehicles"],
        "timing_contract":setup.get("timing_contract","helper_completion_v1"),
        "editor_background_throttle_enabled":setup.get("editor_background_throttle_enabled",True),
        "cpu_placement":setup["cpu_placement"],"source_run":str(source.resolve()),"source_sha256":hashes,
        "runtime_source_sha256":setup["source_sha256"],"costs":costs,
        "normal_parent_activity_cutoff_s":cutoffs,"post_abort_cleanup_visibility_excluded":True,
        "statistic":statistic,
        "method":f"each stage: max(1ms, 1.25*observed {statistic} + 0.5ms); original dispatch/completion guards retained",
        "statistics_s":{k:{"count":len(v),"p95":float(np.percentile(v,95)),"max":max(v)} for k,v in values.items()},
        "sampling_and_command_costs_include_parent_to_helper_queue":True,
        "stages_include_parent_completion_visibility":fields is VISIBLE_FIELDS,
        "is_WCET":False,"holdout_validated":False,"FPGA_applicable":False,
        "soft_prediction_only":True,"tail_coverage_not_guaranteed":True,
        "note":"Empirical development profile. Failed trials retained; no deadlines, B, models or mode-map labels changed."}
    if decompose_frontend_wait:
        result.update(frontend_queue_accounting="recorded_busy_and_forward_command_v2",
            command_forecast_endpoint="rpc_return_s; full state/feedback remains parent-visible via io_s",
            visibility_tail_statistics_s={k:{"count":len(v),"p95":float(np.percentile(v,95)),"max":max(v)}
                for k,v in visible_values.items()},
            residual_policy="pending-stage tail keeps gross observed parent-visible maximum; future forecasts serialize frontend busy work once")
    if output.exists():
        raise ValueError("retain the existing frozen profile; choose a new filename")
    atomic_json(output,result)
    print(json.dumps({"output":str(output),"vehicles":result["vehicles"],"costs":costs},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--statistic",choices=("maximum","p95"),default="maximum")
    parser.add_argument("--decompose-frontend-wait",action="store_true")
    args = parser.parse_args()
    freeze(args.source_run,args.output,args.statistic,args.decompose_frontend_wait)
