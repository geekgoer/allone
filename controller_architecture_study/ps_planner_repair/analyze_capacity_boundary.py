"""Check the frozen capacity trial and summarize per-loop delivery and guard failures."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .critical_update_screen import load, rows
from .prepare_capacity_phase_pair import same_case


def delivered(batch):
    return sum(r["status"] == "delivered" for r in batch)


def p99_ms(us):
    return sorted(us)[math.ceil(.99 * len(us)) - 1] / 1000 if us else None


def local_run(trial, cfg, case, audit, common_us):
    identity = load(trial / "execution_identity.json")
    setup = load(trial / "prepared_source.json")
    origin = identity["origin_us"]
    assert same_case(setup["original_case"], case) and sorted(setup["vehicles"]) == sorted(cfg["names"])
    assert same_case(load(trial / "config_source.json"), cfg)
    assert audit["audit_passed"] and identity["policy"] == "fixed" and not audit["strategy_effect_observed"]
    period = cfg["period_ms"]
    default_service = next(m["service_ms"] for m in identity["initial_modes"] if m["numeric_id"] == cfg["model_id"])
    default_reserve = cfg["command_reserve_ms"]
    expiries = {e["vehicle"]: e for e in load(trial / "hold_expiries.json")}
    by_vehicle = defaultdict(list)
    response, ready_wait, dpu_call, late_returns = [], [], [], []
    busy_ms = 0.
    for row in rows(trial):
        v, cycle = row["vehicle"], row["cycle"]
        phase = cfg["initial_phase_ms"][v]
        assert row["period_ms"] == period and row["model_id"] == cfg["model_id"]
        assert abs(row["planned_release_us"] - origin - 1000 * (phase + cycle * period)) <= 2
        by_vehicle[v].append(row)
        if row["status"] == "delivered":
            response.append(row["command_return_us"] - row["planned_release_us"])
        if row["status"] == "command_return_late":
            late_returns.append(dict(vehicle=v, cycle=cycle,
                release_to_submit_ms=(row["command_submit_us"] - row["planned_release_us"]) / 1000,
                release_to_ack_ms=(row["command_return_us"] - row["planned_release_us"]) / 1000,
                ping_to_ack_ms=(row["command_return_us"] - row["command_ping_start_us"]) / 1000))
        dispatch = row.get("calendar_finished", {})
        if row.get("inference_dispatched") and "board_done_ms" in dispatch:
            lo, hi = dispatch["board_start_ms"], dispatch["board_done_ms"]
            busy_ms += max(0., min(hi, (origin + common_us) / 1000) - max(lo, origin / 1000))
            ready_wait.append(row["calendar_take_start_us"] - row["dpu_enqueue_us"])
            dpu_call.append(row["dpu_local_return_us"] - row["dpu_dequeue_us"])

    per_vehicle, diagnoses = {}, []
    statuses = Counter()
    common_seen = common_delivered = 0
    for v, samples in by_vehicle.items():
        samples.sort(key=lambda r: r["cycle"])
        statuses.update(r["status"] for r in samples)
        common = [r for r in samples if r["planned_release_us"] + 1000 * period <= origin + common_us]
        common_seen += len(common)
        common_delivered += delivered(common)
        acks = sorted(r["command_return_us"] for r in samples if "command_return_us" in r)
        gaps = [b - a for a, b in zip(acks, acks[1:])]
        gates = audit["vehicles"][v]["gate_crossings"]["gates"]
        crossings = [g["first_crossing"] for g in gates if g["first_crossing"]]
        eligible = [c for c in crossings if c["body_valid"] and c["telemetry_gap_acceptable"]]
        expiry = expiries.get(v)
        result = dict(planned=math.ceil((1000 * identity["duration_s"] - cfg["initial_phase_ms"][v]) / period),
                      releases=len(samples), delivered=delivered(samples),
                      max_internal_ack_gap_ms=max(gaps, default=0) / 1000 if gaps else None,
                      valid_gates=len(eligible), expected_gates=len(gates),
                      min_observed_gate_margin_m=min((c["body_margin_m"] for c in crossings), default=None),
                      hold_expiry_s=(expiry["detected_ms"] - origin / 1000) / 1000 if expiry else None,
                      since_last_ack_at_expiry_ms=None)
        if expiry:
            detected = expiry["detected_ms"] * 1000
            last_ack = max(a for a in acks if a <= detected)
            result["since_last_ack_at_expiry_ms"] = (detected - last_ack) / 1000
            for r in samples:
                if not last_ack < r["planned_release_us"] < detected:
                    continue
                evidence = (r.get("calendar_dispatch") or {}).get("dispatch_rejection_evidence")
                diagnostic = dict(vehicle=v, cycle=r["cycle"], status=r["status"], evidence=evidence)
                if evidence:
                    budget = evidence["budget_ms"]
                    required = evidence["required_service_ms"]
                    extra_reserve = evidence["command_reserve_ms"] - default_reserve
                    diagnostic["ready_after_release_ms"] = (r["dpu_enqueue_us"] - r["planned_release_us"]) / 1000
                    diagnostic["ready_to_budget_check_ms"] = evidence["checked_ms"] - r["dpu_enqueue_us"] / 1000
                    # One-row sensitivity using fixed measured timestamps, no new schedule.
                    diagnostic["fits_baseline_model_cost_only"] = default_service <= budget
                    diagnostic["fits_baseline_command_reserve_only"] = required <= budget + extra_reserve
                    diagnostic["fits_both_baseline_costs"] = default_service <= budget + extra_reserve
                diagnoses.append(diagnostic)
        per_vehicle[v] = result

    assert statuses == audit["counts"]["status"]
    return dict(trial=trial.parent.name, period_ms=period, audited=True,
                timing_passed=audit["calendar_timing_acceptance"], flight_passed=audit["complete_flight_observed"],
                collision_events=audit["counts"]["collision_events"],
                final_snapshot_new_collisions=sum(v["final_snapshot_new_collisions"] for v in audit["vehicles"].values()),
                statuses=statuses, vehicles=per_vehicle, retirements=diagnoses, late_returns=late_returns,
                common=dict(duration_ms=common_us / 1000, releases=common_seen, delivered=common_delivered,
                            executor_occupied_fraction=busy_ms / (common_us / 1000)),
                response_p99_ms=p99_ms(response), ready_to_take_p99_ms=p99_ms(ready_wait),
                dpu_call_p99_ms=p99_ms(dpu_call), max_dpu_call_ms=max(dpu_call, default=0) / 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "inputs", "audit", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for line in (args.runs / "inputs.sha256").read_text(encoding="ascii").splitlines():
        digest, filename = line.split(maxsplit=1)
        path = args.inputs / filename
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename
    by_run = {Path(r["data"]).parent.name: r for r in load(args.audit)["runs"]}
    case = load(args.inputs / "case.json")
    trials = sorted(args.runs.glob("*/trial"))
    assert set(by_run) == {f"{arm}_{i}" for arm in ("n12_t20", "n10_t20", "n12_t30") for i in (1, 2)}
    assert {t.parent.name for t in trials} == set(by_run)
    common_us = 32_000_000
    for trial in trials:
        origin = load(trial / "execution_identity.json")["origin_us"]
        for expiry in load(trial / "hold_expiries.json"):
            common_us = min(common_us, round(expiry["detected_ms"] * 1000) - origin)
    result = [local_run(t, load(args.inputs / f"config_{t.parent.name.rsplit('_', 1)[0]}.json"),
                        case, by_run[t.parent.name], common_us) for t in trials]
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
