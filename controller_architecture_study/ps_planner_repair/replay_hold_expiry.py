"""Replay one acknowledged control generation from an original PS flight trace."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def publisher_us(row):
    return round(row["calendar_release"]["published_ms"] * 1_000)


def replay(trial, vehicle, ack_cycle):
    trace = trial / "requests.jsonl"
    config = json.loads((trial / "execution_identity.json").read_text(encoding="utf-8"))["live_config"]
    cap_us = round(config["maximum_update_ms"] * 1_000)
    rows, publisher_ticks = [], set()
    digest = hashlib.sha256()
    with trace.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            publisher_ticks.add(publisher_us(row))
            if row["vehicle"] == vehicle:
                rows.append(row)
    rows.sort(key=lambda row: row["cycle"])
    anchor = next(row for row in rows if row["cycle"] == ack_cycle)
    ack_us = anchor["command_return_us"]
    version = anchor["new_held_version"]
    cap_deadline = ack_us + cap_us
    subsequent = [row for row in rows if row["cycle"] > ack_cycle]

    # Use timestamps after observe() to make the replay causal. Processing a
    # logged sample never makes that sample visible at its acquisition time.
    observations = []
    if anchor.get("paired_observe_accepted") is True:
        sample_us = anchor["paired_sample_start_us"]
        paired_due_us = min(cap_deadline, sample_us + round(anchor["paired_hold_budget_s"] * 1_000_000))
        observations.append((anchor["slot_return_us"], sample_us, paired_due_us))
    for row in subsequent:
        if row.get("observe_accepted") is True and row.get("held_version") == version:
            observations.append((row["feature_ready_us"], row["sample_start_us"], row["hold_deadline_us"]))

    deadline = cap_deadline
    known_expiry_us = deadline
    last_sample_us = ack_us - 1
    for visible_us, sample_us, hold_us in sorted(observations):
        if visible_us >= deadline:
            break  # An expired generation stays expired, even on a later sample.
        if sample_us <= last_sample_us:
            continue
        last_sample_us = sample_us
        deadline = min(cap_deadline, hold_us)
        known_expiry_us = max(visible_us, deadline)
        if visible_us >= deadline:
            break

    ticks = sorted(publisher_ticks)
    cutoff_us = next(tick for tick in ticks if tick >= known_expiry_us)
    # Tie handling between a new ACK and a real safety timer requires an atomic
    # runtime policy. This single-generation replay excludes that case.
    if any(row.get("command_return_us", float("inf")) <= cutoff_us for row in subsequent):
        raise ValueError("another command was acknowledged before or at the replayed exit")
    guard = next(row for row in subsequent if row.get("error") == "flight_guard")
    own_tick_us = next(publisher_us(row) for row in subsequent if publisher_us(row) >= known_expiry_us)
    suppressed = [row for row in subsequent if publisher_us(row) >= cutoff_us]
    historical_expired_at_enqueue = [
        row for row in suppressed
        if (row.get("calendar_queued") or {}).get("cancelled") is True
        and row.get("hold_deadline_us") == deadline
        and row["feature_ready_us"] >= deadline
    ]
    outstanding_rpc = any(
        row.get("command_attempt_start_us", float("inf")) < cutoff_us <= row.get("command_return_us", float("inf"))
        for row in subsequent
    )
    return {
        "scope": "causal trace replay of one generation; logical exit only",
        "trace_sha256": digest.hexdigest(),
        "vehicle": vehicle,
        "ack_cycle": ack_cycle,
        "control_version": version,
        "last_ack_us": ack_us,
        "cap_deadline_us": cap_deadline,
        "observed_hold_deadline_us": deadline,
        "expiry_known_us": known_expiry_us,
        "publisher_first_recorded_tick_at_or_after_expiry_us": cutoff_us,
        "vehicle_first_recorded_release_at_or_after_expiry_us": own_tick_us,
        "simulated_exit_event": {
            "vehicle": vehicle, "control_version": version,
            "reason": "hold_expired", "at_us": cutoff_us,
            "command_in_flight": outstanding_rpc,
        },
        "first_suppressed_cycle": suppressed[0]["cycle"],
        "historical_records_after_exit": dict(Counter(row["status"] for row in suppressed)),
        "historical_expired_hold_rejections_at_enqueue": len(historical_expired_at_enqueue),
        "last_pre_expiry_sample": next(
            {"cycle": row["cycle"], "sample_us": row["sample_start_us"], "tracking_error_m": row["tracking_error_m"]}
            for row in reversed(subsequent)
            if row.get("sample_start_us", float("inf")) < deadline and "tracking_error_m" in row
        ),
        "historical_guard_sample_us": guard["sample_start_us"],
        "historical_guard_cycle": guard["cycle"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--ack-cycle", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.trial, args.vehicle, args.ack_cycle)
    output = json.dumps(report, indent=2) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
