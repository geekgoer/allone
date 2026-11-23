"""Summarize fleet recovery, per-loop deadlines and matched peer work."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from .critical_update_screen import load, rows
from .prepare_capacity_phase_pair import same_case


def analyze(trial, cfg, case, audit):
    identity = load(trial / "execution_identity.json")
    prepared = load(trial / "prepared_source.json")
    final = load(trial / "calendar_final.json")
    origin = identity["origin_us"]
    assert audit["audit_passed"] and identity["policy"] == "fixed" and not identity["enable_skip"]
    assert same_case(prepared["original_case"], case) and same_case(load(trial / "config_source.json"), cfg)
    assert set(prepared["vehicles"]) == set(cfg["names"])
    history = defaultdict(list)
    for r in rows(trial):
        assert r["model_id"] == cfg["model_id"]
        history[r["vehicle"]].append(r)
    expiry = {e["vehicle"]: e for e in load(trial / "hold_expiries.json")}
    probes = {(p["vehicle"], p["cycle"]): p["ack_delay_ms"] for p in cfg["ack_delay_probes"]}
    injections, changes, late, attempts, intervals = [], [], [], [], []
    outcomes, vehicles, rejected_checks = {}, {}, []
    prep_peak = None
    for vehicle, sequence in sorted(history.items()):
        sequence.sort(key=lambda r: r["cycle"])
        assert abs(sequence[0]["planned_release_us"] - origin - 1000*cfg["initial_phase_ms"][vehicle]) <= 2
        assert sequence[0]["period_ms"] == cfg["period_ms"]
        acks = [r["command_return_us"] for r in sequence if "command_return_us" in r]
        for r in sequence:
            cycle = r["cycle"]
            stamp = (r["planned_release_us"]-origin)/1000
            outcomes[vehicle, cycle] = (stamp, r["period_ms"], r["status"])
            if "feature_ready_us" in r:
                start = r["sample_start_us"]
                elapsed = (r["feature_ready_us"]-start)/1000
                if prep_peak is None or elapsed > prep_peak["elapsed_ms"]:
                    prep_peak = dict(vehicle=vehicle, cycle=cycle, elapsed_ms=elapsed,
                                     ready_ms=(r["feature_ready_us"]-origin)/1000,
                                     sample_to_state_return_ms=(r["state_ready_us"]-start)/1000)
            reject = (r.get("calendar_dispatch") or {}).get("dispatch_rejection_evidence")
            if reject:
                rejected_checks.append(reject)
            if "probe_real_ping_return_us" in r:
                delay = (r["command_return_us"]-r["probe_real_ping_return_us"])/1000
                assert delay >= probes[vehicle, cycle]-.002
                injections.append(dict(vehicle=vehicle, cycle=cycle, release_ms=stamp, delay_ms=delay, status=r["status"]))
            if r["status"] == "command_return_late":
                late.append(dict(vehicle=vehicle, cycle=cycle, release_ms=stamp,
                    injected="probe_real_ping_return_us" in r,
                    response_ms=(r["command_return_us"]-r["planned_release_us"])/1000,
                    ping_ms=(r.get("probe_real_ping_return_us", r["command_return_us"])-r["command_ping_start_us"])/1000))
            stage = r.get("calendar_finished", {})
            if r.get("inference_dispatched") and "board_done_ms" in stage:
                intervals.append((stage["board_done_ms"]-origin/1000,
                                  stage["board_done_ms"]-stage["board_start_ms"], vehicle, cycle,
                                  (r["dpu_result_readable_us"]-r["dpu_submit_us"])/1000))
            decision = r.get("command_retime_attempt")
            if decision:
                assert cfg["enable_command_retime"] and r["status"] == "dropped_inference"
                trigger = sequence[decision["late_ack_cycle"]]
                assert trigger["status"] == "command_return_late" and trigger["new_held_version"] == decision["held_version"] == r["held_version"]
                attempt = dict(decision, vehicle=vehicle, cycle=cycle, release_ms=stamp)
                attempts.append(attempt)
                action = r.get("command_retime")
                assert bool(action) == decision["accepted"]
                if action:
                    assert action["cycle"] == cycle+1 and action["commit_ms"] < action["old_release_ms"]
                    next_row = sequence[cycle+1] if cycle+1 < len(sequence) else None
                    if next_row:
                        assert next_row["period_ms"] == action["period_ms"]
                        assert abs(next_row["planned_release_us"]/1000-action["next_release_ms"]) <= .002
                    ack = next_row.get("command_return_us") if next_row else None
                    changes.append(dict(attempt, period_ms=action["period_ms"],
                        original_hold_ms=action["hold_deadline_ms"]-origin/1000,
                        successor_status=next_row["status"] if next_row else None,
                        before_old_hold_ms=action["hold_deadline_ms"]-ack/1000 if ack is not None else None,
                        ack_gap_ms=(ack-trigger["command_return_us"])/1000 if ack is not None else None))
        flight = audit["vehicles"][vehicle]
        gates = flight["gate_crossings"]["gates"]
        end = expiry.get(vehicle)
        rejected = [r for r in sequence if end and acks and r["planned_release_us"]>acks[-1]
                    and (r.get("calendar_dispatch") or {}).get("dispatch_rejection_evidence")]
        vehicles[vehicle] = dict(releases=len(sequence), period_histogram=Counter(r["period_ms"] for r in sequence),
            statuses=Counter(r["status"] for r in sequence), retired=audit["calendar_lifecycle"]["vehicles"][vehicle]["retired"],
            valid_gates=sum(1 for g in gates if (p := g["first_crossing"]) and p["body_valid"] and p["telemetry_gap_acceptable"]),
            expected_gates=len(gates), completed_flight=flight["flight_passed_observed"],
            tracking_rmse_m=flight["tracking"].get("full_window_tracking_rmse_m"),
            max_ack_gap_ms=max((b-a for a,b in zip(acks,acks[1:])), default=0)/1000,
            hold_expiry_s=(end["detected_ms"]-origin/1000)/1000 if end else None,
            final_confirmation_gap_ms=end["detected_ms"]-acks[-1]/1000 if end and acks else None,
            rejected_after_last_ack_count=len(rejected),
            first_rejection_after_last_ack=dict(rejected[0]["calendar_dispatch"]["dispatch_rejection_evidence"],
                cycle=rejected[0]["cycle"], checked_from_origin_ms=rejected[0]["calendar_dispatch"]["dispatch_rejection_evidence"]["checked_ms"]-origin/1000) if rejected else None)
    cap = next(m["service_ms"] for m in identity["initial_modes"] if m["numeric_id"] == cfg["model_id"])
    raises = []
    for done, service, v, cycle, call in sorted(intervals):
        if service > cap:
            cap = service
            raises.append(dict(done_ms=done, service_ms=service, vehicle=v, cycle=cycle, submit_to_readable_ms=call))
    quality = [v["tracking_rmse_m"] for v in vehicles.values() if v["completed_flight"] and v["tracking_rmse_m"] is not None]
    latest_raise = raises[-1] if raises else None
    after_raise = [r for r in rejected_checks if latest_raise and r["checked_ms"]-origin/1000>=latest_raise["done_ms"]]
    return dict(injections=injections, late_returns=late, retime_attempts=attempts, changes=changes, vehicles=vehicles,
        completed_flights=sum(v["completed_flight"] for v in vehicles.values()), flight_passed=audit["complete_flight_observed"],
        timing_passed=audit["calendar_timing_acceptance"], collisions=audit["counts"]["collision_events"]+
        sum(v["final_snapshot_new_collisions"] for v in audit["vehicles"].values()),
        completed_tracking_rmse_median_m=median(quality) if quality else None, actual_dpu_calls=len(intervals),
        model_budget_raises=raises, final_model_service_ms=final["modes"][str(cfg["model_id"])]["service_ms"],
        final_prep_budget_ms=final["config"]["prep_ms"], observed_prep_peak=prep_peak,
        rejections_after_final_model_raise=len(after_raise),
        rejections_using_final_model_budget=sum(abs(r["required_service_ms"]-cap)<.002 for r in after_raise)), outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "inputs", "audit", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for line in (args.runs / "inputs.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        path = args.inputs / name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    reports = {Path(r["data"]).parent.name:r for r in load(args.audit)["runs"]}
    case = load(args.inputs / "case.json")
    configs = {n:load(args.inputs / f"config_{n}.json") for n in reports}
    runs, outcomes = {}, {}
    for name, report in reports.items():
        runs[name], outcomes[name] = analyze(args.runs / name / "trial", configs[name], case, report)
    fixed, retime = (outcomes[n] for n in ("two_ack_fixed20", "two_ack_retime20"))
    matches = []
    cfg = configs["two_ack_fixed20"]
    for probe in cfg["ack_delay_probes"]:
        v = probe["vehicle"]
        start = probe["cycle"]*cfg["period_ms"]+cfg["initial_phase_ms"][v]
        same = {k for k in fixed.keys() & retime.keys() if k[0]!=v and start-.002<=fixed[k][0]<start+220-.002
                and abs(fixed[k][0]-retime[k][0])<=.002 and fixed[k][1]==retime[k][1]}
        matches.append(dict(target=v, matched=len(same), fixed_misses=sum(fixed[k][2]!="delivered" for k in same),
                            retime_misses=sum(retime[k][2]!="delivered" for k in same)))
    natural = {n:r["vehicles"] for n,r in runs.items() if n.startswith("natural_")}
    common = sorted(set.intersection(*(set(v for v,a in group.items() if a["completed_flight"] and
                                        a["tracking_rmse_m"] is not None) for group in natural.values())))
    rmse = {n:median(group[v]["tracking_rmse_m"] for v in common) for n,group in natural.items()} if common else {}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(dict(runs=runs, matched_peer_windows=matches, natural_completed_common=common,
                       natural_common_tracking_rmse_median_m=rmse), out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
