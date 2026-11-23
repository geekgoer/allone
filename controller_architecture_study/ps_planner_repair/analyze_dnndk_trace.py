"""Join PS command rows with the optional DNNDK call trace.

Use the call interval on the same board monotonic clock. Startup/warmup
calls have no control-job row and are excluded. Driver elapsed time can
overlap with both caller CPU time and runnable-queue wait.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .analyze_dispatch_timing import stats


def analyze(trial, timing):
    with timing.open(newline="", encoding="utf-8") as stream:
        calls = sorted(({key: int(value) for key, value in row.items()}
                        for row in csv.DictReader(stream)), key=lambda row: row["begin_ns"])
    starts = [row["begin_ns"] for row in calls]
    matched = set()
    samples = []
    missing = 0
    with (trial / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            job = json.loads(line)
            if not job.get("inference_dispatched") or job.get("dpu_status") != 0:
                continue
            # Logged PS interval uses microseconds truncated from CLOCK_MONOTONIC.
            submit = job["dpu_submit_us"] * 1000
            returned = (job["dpu_result_readable_us"] + 1) * 1000
            index = bisect_left(starts, submit)
            if (index >= len(calls) or calls[index]["end_ns"] >= returned or
                    calls[index]["status"] != 0 or calls[index]["begin_ns"] > calls[index]["end_ns"]):
                missing += 1
                continue
            if (index in matched or
                    (index + 1 < len(calls) and calls[index + 1]["end_ns"] < returned)):
                raise ValueError("DNNDK trace cannot be matched uniquely")
            matched.add(index)
            call = calls[index]
            wall = (call["end_ns"] - call["begin_ns"]) / 1000.
            cpu = call["thread_cpu_ns"] / 1000.
            driver = call["driver_profile_us"]
            runqueue = call["runqueue_ns"] / 1000.
            outer = job["dpu_result_readable_us"]-job["dpu_submit_us"]
            around = max(0., outer-wall)
            # Proc reads bracket the inner call. Subtract the whole surrounding
            # bracket plus one microsecond for the outer timestamp truncation.
            in_call_wait = max(0., runqueue-around-1.) if 0 <= runqueue <= wall else None
            samples.append(dict(vehicle=job["vehicle"], cycle=job["cycle"], model=job["model_id"],
                                status=job["status"], wall_us=wall, thread_cpu_us=cpu,
                                driver_us=driver, outside_driver_us=wall-driver if 0 < driver <= wall else None,
                                runqueue_us=runqueue if 0 <= runqueue <= wall else None,
                                runqueue_in_call_at_least_us=in_call_wait,
                                voluntary_cs=call["voluntary_cs"], involuntary_cs=call["involuntary_cs"],
                                tid=call["tid"], submit_us=job["dpu_submit_us"], wall_outer_us=outer,
                                around_call_us=around))
    if missing or not samples:
        raise ValueError(f"DNNDK trace coverage incomplete: matched={len(samples)}, missing={missing}")

    def summarize(rows):
        evidence = Counter()
        for row in rows:
            if row["outside_driver_us"] is None:
                evidence["driver_profile_unusable"] += 1
            elif row["outside_driver_us"] > row["driver_us"]:
                evidence["outside_driver_majority"] += 1
            else:
                evidence["driver_interval_majority"] += 1
            if row["runqueue_us"] is None:
                evidence["runqueue_unusable"] += 1
            else:
                evidence["runqueue_positive"] += row["runqueue_us"] > 0
                evidence["runqueue_majority"] += row["runqueue_in_call_at_least_us"] > row["wall_us"] / 2
            evidence["involuntary_switch_seen"] += row["involuntary_cs"] > 0
        fields = ("wall_us", "driver_us", "outside_driver_us", "thread_cpu_us", "runqueue_us",
                  "runqueue_in_call_at_least_us",
                  "voluntary_cs", "involuntary_cs", "around_call_us")
        return {"count": len(rows), "composition": evidence, **{
            field: stats([row[field] for row in rows if row[field] is not None and row[field] >= 0])
            for field in fields}}

    by_model = defaultdict(list)
    for row in samples:
        by_model[row["model"]].append(row)
    return {"matched_calls": len(samples), "calls_without_successful_control_row": len(calls)-len(samples),
            "executor_tids": sorted({row["tid"] for row in samples}), "all": summarize(samples),
            "by_model": {key: summarize(group) for key, group in by_model.items()},
            "tail_threshold_us": {threshold: summarize([row for row in samples if row["wall_us"] >= threshold])
                                  for threshold in (3000, 5000, 10000)},
            "top_slow_calls": sorted(samples, key=lambda row: -row["wall_us"])[:30]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--api-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.trial, args.api_trace)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
