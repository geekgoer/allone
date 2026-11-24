"""Check matching motor outputs and compare the command timestamps in a probe CSV."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


SCENARIOS = ("normal", "delay_before_goal", "delay_after_ping")
FIELDS = ("callback_ms", "goal_set_ms", "motor_update_ms", "ping_return_ms", "reported_ack_ms")


def summarize(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    expected = {(scenario, str(i)) for scenario in SCENARIOS for i in range(1, 13)}
    if len(records) != len(expected) or {(r["scenario"], r["repeat"]) for r in records} != expected:
        raise ValueError("Incomplete or duplicate paired samples")
    if len({r["throttle"] for r in records}) != len(records):
        raise ValueError("Ambiguous command identity")

    groups = {name: [] for name in SCENARIOS}
    for row in records:
        ticks = {field: float(row[field]) for field in FIELDS}
        callback, goal, motor, ping, reported = (ticks[key] for key in FIELDS)
        if not (all(math.isfinite(t) for t in ticks.values()) and 0 <= callback <= goal <= motor
                and 0 <= ping <= reported):
            raise ValueError(f"Invalid timestamps for {row['scenario']} repeat {row['repeat']}")
        throttle = float(row["throttle"])
        if not all(abs(float(row[f"motor_{i}"]) - throttle) <= 1e-5 for i in range(4)):
            raise ValueError(f"Unmatched motor output for throttle {throttle}")
        ticks["callback_to_goal_ms"] = goal - callback
        ticks["goal_to_motor_ms"] = motor - goal
        ticks["observed_after_ping_ms"] = reported - ping
        ticks["reported_minus_motor_ms"] = reported - motor
        groups[row["scenario"]].append(ticks)

    report = {name: dict(count=len(rows), median_ms={k: median(r[k] for r in rows) for k in rows[0]},
                         raw_ping_before_goal=sum(r["ping_return_ms"] < r["goal_set_ms"] for r in rows),
                         raw_ping_before_motor=sum(r["ping_return_ms"] < r["motor_update_ms"] for r in rows),
                         motor_by_20_ms=sum(r["motor_update_ms"] <= 20 for r in rows),
                         reported_ack_by_20_ms=sum(r["reported_ack_ms"] <= 20 for r in rows))
              for name, rows in groups.items()}
    control_targets = {"delay_before_goal": "callback_to_goal_ms", "delay_after_ping": "observed_after_ping_ms"}
    for name, field in control_targets.items():
        if min(r[field] for r in groups[name]) < 39.5:
            raise ValueError(f"Configured 40 ms did not appear in {field}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
