"""Audit where paired experimental jobs lose their pre-command time margin."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from .analyze_dispatch_timing import stats


def load_trial(trial, horizon_us):
    identity = json.loads((trial / "execution_identity.json").read_text(encoding="utf-8"))
    origin = identity["origin_us"]
    reserve = round(identity["live_config"]["command_reserve_ms"] * 1000)
    rows, occupied = {}, []
    sha = hashlib.sha256()
    first_raise = None
    with (trial / "requests.jsonl").open("rb") as stream:
        for line in stream:
            sha.update(line)
            row = json.loads(line)
            sample = row.get("paired_sample_start_us")
            if sample and row["command_return_us"] - row["control_converted_us"] > reserve:
                if first_raise is None or sample - origin < first_raise:
                    first_raise = sample - origin
            if row["planned_release_us"] < origin + horizon_us:
                key = row["vehicle"], row["cycle"]
                if key in rows:
                    raise ValueError(f"Repeated release {key}")
                rows[key] = row
            if row.get("inference_dispatched"):
                start = round(row["calendar_dispatch"]["board_start_ms"] * 1000)
                stop = row["dpu_local_return_us"]
                occupied.append((start, stop))
    occupied.sort()
    if any(end > start for (_, end), (start, _) in zip(occupied, occupied[1:])):
        raise ValueError("Overlapping assignments on the single-DPU trace")
    if first_raise is not None and first_raise <= horizon_us:
        raise ValueError("Choose a paired horizon over which the command reserve is constant")
    return rows, occupied, origin, reserve, first_raise, sha.hexdigest()


def classifier(row, reserve):
    result = row["output_ready_us"]
    cutoff = row["actuation_deadline_us"] - reserve
    if row["calendar_after_dpu"].get("cancelled"):
        return "cancelled"
    if row.get("command_guard_accepted") is False:
        return "final_guard_rejected"
    if result >= cutoff:
        return "short_when_worker_received_result"
    if row["slot_return_us"] >= cutoff:
        return "short_after_worker_received_result_before_precheck"
    return "unexplained_by_fixed_reserve"


def timing(row, occupied, ends, reserve):
    start, take = (row[k] for k in ("dpu_enqueue_us", "calendar_take_start_us"))
    queue = take - start
    index = bisect_right(ends, start)
    busy = 0
    while index < len(occupied) and occupied[index][0] < take:
        lo, hi = occupied[index]
        busy += min(take, hi) - max(start, lo)
        index += 1
    stamp = row["planned_release_us"]
    dequeue = row["dpu_dequeue_us"]
    local_return, output = (row[key] for key in ("dpu_local_return_us", "output_ready_us"))
    stages = {"release_to_DPU_queue": start - stamp, "queue_total": queue,
              "queue_with_other_DPU_assignment": busy, "queue_outside_DPU_assignments": queue - busy,
              "take_to_dequeue": dequeue - take, "local_inference_including_IO": local_return - dequeue,
              "DPU_submit_to_readable": row["dpu_result_readable_us"] - row["dpu_submit_us"],
              "result_handoff": output - local_return, "release_to_result": output - stamp,
              "margin_over_precheck_reserve_at_result": row["actuation_deadline_us"] - output - reserve}
    return stages


def run(reference, candidate, window_us):
    a, ao, ta, ar, araise, asha = load_trial(reference, window_us)
    b, bo, tb, br, braise, bsha = load_trial(candidate, window_us)
    if a.keys() != b.keys() or ar != br:
        raise ValueError("Paired releases and constant reserve must match")
    if any(a[k]["planned_release_us"] - ta != b[k]["planned_release_us"] - tb for k in a):
        raise ValueError("Relative release calendars differ")
    ae, be = [end for _, end in ao], [end for _, end in bo]
    transitions = Counter(f"{a[k]['status']} -> {b[k]['status']}" for k in a)
    details, model_pairs = [], Counter()
    for key, left in a.items():
        right = b[key]
        if left["status"] != "delivered" or right["status"] != "dropped_before_command":
            continue
        x, y = timing(left, ao, ae, ar), timing(right, bo, be, br)
        from_model, to_model = left["model_id"], right["model_id"]
        model_pairs[f"{from_model} -> {to_model}"] += 1
        details.append({"vehicle": key[0], "cycle": key[1], "reference_model": from_model,
                        "candidate_model": to_model, "candidate_class": classifier(right, br),
                        "reference_us": x, "candidate_us": y,
                        "delta_us": {stage: y[stage] - x[stage] for stage in x}})
    if not details:
        raise ValueError("No reference deliveries lost before command in candidate")
    def describe(values):
        return {"mean": mean(values), **stats(values)}
    summary = {kind: {stage: describe([entry[kind][stage] for entry in details]) for stage in details[0][kind]}
               for kind in ("reference_us", "candidate_us", "delta_us")}
    return {"paired_releases": len(a), "window_us": window_us, "reserve_us": ar,
            "first_reserve_increase_proxy_us": [araise, braise], "trace_sha256": [asha, bsha],
            "status_transitions": transitions,
            "lost_vs_delivered_classification": Counter(item["candidate_class"] for item in details),
            "model_pairs": model_pairs, "stages": summary, "matched_losses": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--window-us", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.reference, args.candidate, args.window_us)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
