"""Read-only upper bound on all-loop paired-context availability."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    rows = defaultdict(list)
    with (args.run / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            rows[item["vehicle"]].append(item)
    intervals = {}
    for vehicle, records in rows.items():
        records.sort(key=lambda row: row["cycle"])
        spans, start = [], None
        for row in records:
            if row.get("command_guard_accepted") and start is not None:
                spans.append((start, row["command_guard_start_us"]))
                start = None
            if row.get("paired_observe_accepted"):
                start = row["paired_analysis_ready_us"]
        if start is not None:
            spans.append((start, float("inf")))
        intervals[vehicle] = spans
    starts = {name: [a for a, _ in spans] for name, spans in intervals.items()}
    ready = Counter()
    actions, installs = Counter(), Counter()
    plans = 0
    with (args.run / "plans.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            plan = json.loads(line)
            when = plan["start_us"]
            count = 0
            for name, spans in intervals.items():
                i = bisect_right(starts[name], when) - 1
                count += i >= 0 and when < spans[i][1]
            ready[count] += 1
            actions[plan.get("decision", {}).get("action")] += 1
            installs[plan.get("install", {}).get("status")] += 1
            plans += 1
    report = {
        "scope": "necessary condition only: accepted paired state exists and no command attempt is pending; excludes hold expiry, release and deadline guards",
        "vehicles": len(intervals), "plans": plans,
        "paired_ready_vehicle_count_at_plan_start": dict(sorted(ready.items())),
        "maximum_simultaneously_ready": max(ready) if ready else 0,
        "decision_actions": dict(actions), "install_statuses": dict(installs),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
