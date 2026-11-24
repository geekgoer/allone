"""Screen onboard workload splits using a natural command-retime trial."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .critical_update_screen import load, rows


def period_windows(row, periods, fresh_ms, stop_ms):
    """Available time after a future release, using the logged hold deadline."""
    previous = row["planned_release_us"] / 1000
    observed_hold = row["hold_deadline_us"] / 1000
    return {
        period: min(period, fresh_ms, observed_hold - previous - period, stop_ms - previous - period)
        for period in sorted(periods)
        if period > row["period_ms"]
    }


def analyze(trial):
    config = load(trial / "config_source.json")
    if not config["enable_command_retime"] or config["ack_delay_probes"]:
        raise ValueError("Use the natural retime trial without injected ACK delays")
    stop = load(trial / "calendar_final.json")["config"]["measurement_stop_ms"]
    log = list(rows(trial))
    # Each included observation reached the end of the frontend preparation path.
    prepared = sorted((r for r in log if "feature_ready_us" in r), key=lambda r: r["feature_ready_us"])
    attempts = sorted((r for r in log if "command_retime_attempt" in r), key=lambda r: r["slot_return_us"])
    late = [r for r in log if r["status"] == "command_return_late"]
    if not prepared or not attempts or not late:
        raise ValueError("Trial must contain preparation samples, retime attempts and late ACKs")

    # Each individual post budget is a monotone maximum in the recorded program.
    acks = iter(sorted((r for r in log if "command_return_us" in r), key=lambda r: r["command_return_us"]))
    pending_ack = next(acks, None)

    prep_floor = config["prep_ms"]
    post_floor = max(config["post_ms"], config["command_reserve_ms"])
    prep = through_collision = prep_floor
    post = defaultdict(lambda: post_floor)
    prepared_iter = iter(prepared)
    pending = next(prepared_iter, None)
    report = []
    for row in attempts:
        now = row["slot_return_us"]
        vehicle = row["vehicle"]
        while pending is not None and pending["feature_ready_us"] <= now:
            sample = pending["sample_start_us"]
            prep = max(prep, (pending["feature_ready_us"] - sample) / 1000)
            through_collision = max(through_collision, (pending["collision_ready_us"] - sample) / 1000)
            pending = next(prepared_iter, None)
        while pending_ack is not None and pending_ack["command_return_us"] <= now:
            ack_vehicle = pending_ack["vehicle"]
            elapsed = pending_ack["command_return_us"] - pending_ack["output_ready_us"]
            post[ack_vehicle] = max(post[ack_vehicle], elapsed / 1000)
            pending_ack = next(acks, None)

        evidence = row["calendar_dispatch"]["dispatch_rejection_evidence"]
        model = evidence["required_service_ms"]
        observed_post = max(post[vehicle], evidence["command_reserve_ms"])
        windows = period_windows(row, config["periods_ms"], config["fresh_ms"], stop)
        cases = {
            "recorded": (prep, observed_post),
            "zero_compute_after_collision_rpc": (through_collision, observed_post),
            "short_preparation": (prep_floor, observed_post),
            "short_command": (prep, post_floor),
            "short_preparation_and_command": (prep_floor, post_floor),
        }
        report.append(dict(
            vehicle=vehicle, cycle=row["cycle"], actually_installed=row["command_retime_attempt"]["accepted"],
            model_ms=model, observed_reserve_ms=evidence["command_reserve_ms"],
            period_windows_ms=windows,
            cases={name: dict(preparation_ms=front, post_ms=tail, total_ms=front + model + tail,
                              first_period_ms=next((p for p, window in windows.items() if front + model + tail < window), None))
                   for name, (front, tail) in cases.items()},
        ))

    peak = max(prepared, key=lambda r: r["feature_ready_us"] - r["sample_start_us"])
    late_split = []
    for row in late:
        output = row["output_ready_us"]
        deadline = row["command_admission"]["deadline_us"]
        late_split.append(dict(vehicle=row["vehicle"], cycle=row["cycle"],
                               release_to_output_ms=(output - row["planned_release_us"]) / 1000,
                               output_to_proxy_ms=(row["command_return_us"] - output) / 1000,
                               ping_ms=(row["command_return_us"] - row["command_ping_start_us"]) / 1000,
                               max_post_before_deadline_ms=(deadline - output) / 1000,
                               short_post_would_fit=output + 1000 * post_floor < deadline))
    counts = {case: sum(r["cases"][case]["first_period_ms"] is not None for r in report)
              for case in report[0]["cases"]}
    return dict(scope="Recorded deadlines and service, local window screen only, no calendar commit or flight prediction",
                trial=str(trial), releases=len(log), complete_preparations=len(prepared),
                preparation_peak=dict(vehicle=peak["vehicle"], cycle=peak["cycle"],
                                      to_state_ms=(peak["state_ready_us"] - peak["sample_start_us"]) / 1000,
                                      state_to_collision_ms=(peak["collision_ready_us"] - peak["state_ready_us"]) / 1000,
                                      after_collision_ms=(peak["feature_ready_us"] - peak["collision_ready_us"]) / 1000,
                                      total_ms=(peak["feature_ready_us"] - peak["sample_start_us"]) / 1000),
                late_commands=late_split, attempts=report, windows_passed=counts,
                actual_installations=sum(r["actually_installed"] for r in report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.trial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result["windows_passed"]))


if __name__ == "__main__":
    main()
