"""Separate PS dispatch, local accelerator service and command timing in a PS trial."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def stats(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(ordered), "p50": ordered[len(ordered) // 2],
            "p99": ordered[int(.99 * (len(ordered) - 1))], "max": ordered[-1]}


def analyze(trial):
    identity = json.loads((trial / "execution_identity.json").read_text(encoding="utf-8"))
    runtime = json.loads((trial / "summary.json").read_text(encoding="utf-8"))
    final = json.loads((trial / "calendar_final.json").read_text(encoding="utf-8"))
    expiries = trial / "hold_expiries.json"
    events = json.loads(expiries.read_text(encoding="utf-8")) if expiries.exists() else []
    stages, per_model = defaultdict(list), defaultdict(lambda: defaultdict(list))
    statuses, per_vehicle, dominant = Counter(), defaultdict(Counter), Counter()
    rejection_evidence, worst, missing, rejection_reasons = [], [], Counter(), Counter()
    rejection_shortfalls = defaultdict(list)
    slow_lock_windows = []
    before_command = Counter()
    preliminary_budget_loss = Counter()
    replay_headroom = []
    admissions = Counter()
    service_events, rejection_events = [], []
    stamped = downgraded_at_handoff = 0
    trace_sha = hashlib.sha256()
    with (trial / "requests.jsonl").open("rb") as stream:
        for line in stream:
            trace_sha.update(line)
            row = json.loads(line)
            status = row["status"]
            statuses[status] += 1
            per_vehicle[row["vehicle"]][status] += 1
            dispatch = row.get("calendar_dispatch") or {}
            if dispatch.get("dispatch_rejection_evidence"):
                evidence = dispatch["dispatch_rejection_evidence"]
                rejection_evidence.append(evidence)
                reason = dispatch["dispatch_rejection"]
                rejection_reasons[reason] += 1
                rejection_shortfalls[reason].append(round(1000 * (evidence["required_service_ms"] - evidence["budget_ms"])))
                if row.get("dpu_dequeue_us") is not None:
                    rejection_events.append((row["dpu_dequeue_us"], evidence))
            ack = row.get("command_return_us")
            if ack is not None and row.get("control_converted_us") is not None:
                stages["command_prepare_to_ack"].append(ack - row["control_converted_us"])
            admission = row.get("command_admission")
            if admission is not None:
                accepted = admission["accepted"]
                if accepted != row["command_guard_accepted"]:
                    raise ValueError("Atomic admission disagrees with the command guard")
                slack = admission["deadline_us"] - admission["reserve_us"] - admission["checked_us"]
                admissions["accepted" if accepted else admission["rejection"]] += 1
                stages["command_guard_lock_wall"].append(admission["lock_wall_us"])
                stages["command_guard_slack"].append(slack)
                stages["command_output_to_guard_check"].append(admission["checked_us"] - row["output_ready_us"])
                if accepted and slack < -0.001:
                    raise ValueError("Command accepted without sufficient reserve")
                if status in ("delivered", "command_return_late"):
                    if not accepted or (abs(ack - admission["deadline_us"]) > 0.001 and
                                        (ack < admission["deadline_us"]) != (status == "delivered")):
                        raise ValueError("ACK classification disagrees with the effective deadline")
            precheck = row.get("command_precheck")
            if precheck is not None:
                output = row["output_ready_us"]
                start, end, checked, reserve_read = (precheck[key] for key in
                    ("row_start_us", "row_return_us", "checked_us", "reserve_return_us"))
                lock = precheck["row_lock_wall_us"]
                if not output <= start <= end <= checked <= reserve_read:
                    raise ValueError("Nonmonotonic command precheck stamps")
                split_precheck = {"precheck_output_to_row": start - output, "precheck_row_lock_wall": lock,
                                  "precheck_row_other_wall": max(0., end - start - lock),
                                  "precheck_row_return_to_check": checked - end,
                                  "precheck_reserve_read": reserve_read - checked}
                for stage, elapsed in split_precheck.items():
                    stages[stage].append(elapsed)
                cutoff = precheck["deadline_us"] - precheck["reserve_us"]
                if not precheck["accepted"]:
                    if precheck["cancelled"]:
                        cause = "cancelled"
                    else:
                        assert checked >= cutoff
                        # Keep the measured first lock wait and the old cutoff. This
                        # screens potential headroom, without replaying contention,
                        # wire conversion, or changes to the active hold deadline.
                        replay_headroom.append(cutoff - (start + lock))
                        marks = (output, start, start + lock, end, checked)
                        names = ("already_short_at_worker_output", "output_to_row", "row_lock_wall",
                                 "row_other_wall", "row_return_to_check")
                        cause = next(name for t, name in zip(marks, names) if t >= cutoff)
                    preliminary_budget_loss[cause] += 1
            if status == "dropped_before_command" and row.get("output_ready_us") is not None:
                if row["output_ready_us"] >= row["actuation_deadline_us"]:
                    before_command["result_ready_at_or_after_deadline"] += 1
                elif row.get("command_guard_accepted") is False:
                    before_command["result_before_deadline_then_final_guard_rejected"] += 1
                else:
                    before_command["result_before_deadline_then_preliminary_check_rejected"] += 1
            if not row.get("inference_dispatched"):
                continue
            downgraded_at_handoff += "dispatch_handoff_downgraded_from" in dispatch
            required = ("calendar_take_start_us", "calendar_take_return_us", "calendar_take_lock_wait_us",
                        "dpu_enqueue_us", "dpu_dequeue_us", "dpu_submit_us", "dpu_result_readable_us", "dpu_local_return_us")
            absent = [field for field in required if row.get(field) is None]
            if absent:
                missing.update(absent)
                continue
            enter, leave, lock, enqueue, dequeue, submit, output, returned = (row[key] for key in required)
            if not enqueue <= enter <= leave <= dequeue <= submit <= output <= returned:
                missing["nonmonotonic_dispatch_stamps"] += 1
                continue
            # Lock duration uses a higher resolution than the surrounding stamps.
            split = {"take_lock_wall": lock, "take_other_wall": max(0., leave - enter - lock),
                     "take_return_to_dequeue": dequeue - leave, "dnndk_input_to_submit": submit - dequeue,
                     "dnndk_submit_to_readable": output - submit, "dnndk_readable_to_return": returned - output}
            own_since = dispatch.get("board_start_ms", enter / 1_000) * 1_000
            values = {**split, "ready_to_take": enter - enqueue,
                      "take_wall": leave - enter, "take_to_local_return": returned - enter,
                      "dequeue_to_local_return": returned - dequeue,
                      "charged_model_service": round(returned - own_since)}
            for key, value in values.items():
                stages[key].append(value)
                per_model[row["model_id"]][key].append(value)
            stamped += 1
            service_events.append((returned, row["model_id"], returned - dequeue))
            if returned - enter >= 3000:
                largest = max(split, key=split.get)
                dominant[largest] += 1
                if largest == "take_lock_wall":
                    slow_lock_windows.append((enter, enter + lock))
                worst.append({"vehicle": row["vehicle"], "cycle": row["cycle"], "model": row["model_id"],
                              "status": status, "largest_stage": largest, "timing_us": values})
    if not stamped:
        raise ValueError("No successful timestamped dispatch found")
    reserve = identity["live_config"]["command_reserve_ms"]
    model_cost = {row["numeric_id"]: row["service_ms"] * 1000 for row in identity["initial_modes"]}
    ids = identity["live_config"]["candidate_ids"]
    # Offline shadow only. Keep the original empirical floor and grow each
    # estimate with previously observed local service, independent of PS take.
    services = iter(sorted(service_events))
    following = next(services, None)
    shadow = Counter()
    for now, evidence in sorted(rejection_events, key=lambda event: event[0]):
        while following and following[0] < now:
            _, model, cost = following
            model_cost[model] = max(model_cost[model], cost)
            following = next(services, None)
        candidates = ids if identity["policy"] == "adaptive" else [evidence["required_model_id"]]
        available_us = 1000 * (evidence["checked_ms"] + evidence["budget_ms"]) - now
        shadow["rejections_checked"] += 1
        shadow["would_fit_at_actual_handoff"] += min(model_cost[model] for model in candidates) <= available_us
    planner_snapshot_spans = []
    snapshot_builds = defaultdict(list)
    installed = 0
    with (trial / "plans.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            plan = json.loads(line)
            installed += bool((plan.get("install") or {}).get("installed"))
            timing = (plan.get("management_timing") or {}).get("snapshot") or {}
            wait, build = (timing.get(stage, {}).get("wall_ms") for stage in ("lock_wait", "build"))
            if wait is not None and build is not None:
                usable = plan.get("initial_snapshot") or plan.get("snapshot")
                snapshot_builds["usable" if usable is not None else "unusable"].append(round(build * 1000))
                start = plan["start_us"] + wait * 1_000
                planner_snapshot_spans.append((start, start + build * 1_000))
    overlaps = sum(any(start < other_end and other_start < end for other_start, other_end in planner_snapshot_spans)
                   for start, end in slow_lock_windows)
    return {"historical_run": str(trial), "trace_sha256": trace_sha.hexdigest(), "statuses": statuses,
            "vehicle_statuses": per_vehicle, "runtime": runtime, "expiries": events,
            "stamp_rows": stamped, "missing_dispatch_fields": missing,
            "stages_us": {k: stats(v) for k, v in stages.items()},
            "service_by_model_us": {model: {k: stats(v[k]) for k in ("take_to_local_return", "dequeue_to_local_return",
                                                                    "dnndk_submit_to_readable")}
                                    for model, v in per_model.items()},
            "dominant_in_instances_over_3ms": dominant, "planner_installs": installed,
            "planner_snapshot_build_us": {k: stats(v) for k, v in snapshot_builds.items()},
            "before_command_drops": before_command,
            "precheck_cutoff_crossings": preliminary_budget_loss,
            "atomic_command_admissions": admissions,
            "historical_after_first_lock_screen": {
                "noncancelled_precheck_drops": len(replay_headroom),
                "remaining_positive_us_after_assumed_extra_us": {
                    extra: sum(headroom > extra for headroom in replay_headroom)
                    for extra in (0, 50, 100, 250)
                },
            },
            "slow_lock_overlap_with_approx_planner_snapshot": {"n": len(slow_lock_windows), "overlap": overlaps},
            "top_slow_services": sorted(worst, key=lambda d: -d["timing_us"]["take_to_local_return"])[:20],
            "rejection_evidence_count": len(rejection_evidence),
            "rejection_reasons": rejection_reasons,
            "rejection_shortfall_us_by_reason": {k: stats(v) for k, v in rejection_shortfalls.items()},
            "downgraded_at_handoff": downgraded_at_handoff,
            "rejection_required_model_counts": Counter(r["required_model_id"] for r in rejection_evidence),
            "rejection_required_ms": stats([r["required_service_ms"] for r in rejection_evidence]),
            "rejection_budget_ms": stats([r["budget_ms"] for r in rejection_evidence]),
            "offline_separate_budget_screen": shadow, "final_local_service_costs_us": model_cost,
            "initial_command_reserve_ms": reserve, "final_command_reserve_ms": final["config"]["command_reserve_ms"],
            "final_model_costs_ms": {m: v["service_ms"] for m, v in final["modes"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trial)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
