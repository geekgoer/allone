"""Audit a matched slow-ACK pilot and quantify recovery and peer impact."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .critical_update_screen import load, rows
from .prepare_capacity_phase_pair import same_case


def local_run(trial, config, case, audit):
    identity = load(trial / "execution_identity.json")
    setup = load(trial / "prepared_source.json")
    assert audit["audit_passed"] and identity["policy"] == "fixed" and not identity["enable_skip"]
    assert same_case(setup["original_case"], case) and same_case(load(trial / "config_source.json"), config)
    assert sorted(setup["vehicles"]) == sorted(config["names"])
    assert set(audit["calendar_lifecycle"]["actual_inference_model_histogram"]) == {str(config["model_id"])}
    origin = identity["origin_us"]
    probe = config["command_retime_probe"]
    target = probe["vehicle"]
    grouped = defaultdict(list)
    for request in rows(trial):
        grouped[request["vehicle"]].append(request)
    holds = {e["vehicle"]: e for e in load(trial / "hold_expiries.json")}
    commands, peers, late_returns = {}, {}, []
    action = None
    injection = None
    target_recovery = None
    for name, sample in sorted(grouped.items()):
        sample.sort(key=lambda r: r["cycle"])
        acks = sorted(r["command_return_us"] for r in sample if "command_return_us" in r)
        for r in sample:
            if r["status"] == "command_return_late":
                late_returns.append(dict(vehicle=name, cycle=r["cycle"], injected="probe_real_ping_return_us" in r,
                    release_s=(r["planned_release_us"]-origin)/1e6,
                    release_to_submit_ms=(r["command_submit_us"]-r["planned_release_us"])/1000,
                    ping_to_ack_ms=(r.get("probe_real_ping_return_us",r["command_return_us"])-r["command_ping_start_us"])/1000,
                    release_to_ack_ms=(r["command_return_us"]-r["planned_release_us"])/1000))
        gates = audit["vehicles"][name]["gate_crossings"]["gates"]
        valid = [p for g in gates if (p := g["first_crossing"]) and p["body_valid"] and p["telemetry_gap_acceptable"]]
        stat = Counter(r["status"] for r in sample)
        loop = audit["calendar_lifecycle"]["vehicles"][name]
        expiry = holds.get(name)
        tail = [r for r in sample if expiry and acks and acks[-1]<r["planned_release_us"]<expiry["detected_ms"]*1000]
        commands[name] = dict(statuses=stat, elapsed_until_hold_expiry_s=(expiry["detected_ms"]*1000-origin)/1e6 if expiry else None,
                              expiry_gap_ms=(expiry["detected_ms"]*1000-acks[-1])/1000 if expiry and acks else None,
                              rejected_before_expiry=[dict(cycle=r["cycle"], status=r["status"], evidence=(r.get("calendar_dispatch") or {}).get("dispatch_rejection_evidence")) for r in tail],
                              max_ack_gap_ms=max((b-a for a, b in zip(acks, acks[1:])), default=0)/1000,
                              valid_gates=len(valid), expected_gates=len(gates), retired=loop["retired"],
                              released_periods=loop["released_period_histogram_ms"])
        if name != target:
            assert all(r["period_ms"] == config["period_ms"] and "command_retime" not in r and
                       "probe_real_ping_return_us" not in r for r in sample)
            peers.update({(name, r["cycle"]): r["status"] for r in sample})
            continue
        injected = [r for r in sample if "probe_real_ping_return_us" in r]
        updates = [r for r in sample if "command_retime" in r]
        assert len(injected) <= 1 and len(updates) <= int(probe["retime"])
        if injected:
            r = injected[0]
            assert r["cycle"] == probe["cycle"] and r["status"] == "command_return_late"
            injection = dict(cycle=r["cycle"], delay_ms=(r["command_return_us"]-r["probe_real_ping_return_us"])/1000,
                             release_to_wire_ack_ms=(r["probe_real_ping_return_us"]-r["planned_release_us"])/1000,
                             release_to_observed_ack_ms=(r["command_return_us"]-r["planned_release_us"])/1000)
            assert injection["delay_ms"] >= probe["ack_delay_ms"] - .002
            target_recovery = dict(next_delivered_cycle=None, gap_from_injected_ack_ms=None, old_hold_slack_ms=None)
            after = next((q for q in sample if q["cycle"] > r["cycle"] and q["status"] == "delivered"), None)
            if after:
                target_recovery.update(next_delivered_cycle=after["cycle"],
                                       gap_from_injected_ack_ms=(after["command_return_us"]-r["command_return_us"])/1000)
        if updates:
            update = updates[0]
            action = update["command_retime"]
            assert injection and update["held_version"] == injected[0]["new_held_version"]
            assert action["cycle"] == update["cycle"]+1 and action["commit_ms"] < action["old_release_ms"]
            successor = next((r for r in sample if r["cycle"] == action["cycle"]), None)
            if successor:
                assert successor["period_ms"] == action["period_ms"] and abs(successor["planned_release_us"]/1000-action["next_release_ms"]) <= .002
            action["successor_status"] = successor["status"] if successor else None
            action["release_change_ms"] = action["next_release_ms"] - action["old_release_ms"]
            action["commit_before_original_release_ms"] = action["old_release_ms"] - action["commit_ms"]
            if after:
                target_recovery["old_hold_slack_ms"] = action["hold_deadline_ms"]-after["command_return_us"]/1000
        assert bool(action) == any(r["period_ms"] != config["period_ms"] for r in sample)
    result = dict(run=trial.parent.name, target=target, origin_us=origin, injection=injection,
                  action=action, target_recovery=target_recovery, vehicles=commands, late_returns=late_returns,
                  audit_passed=True, flight_passed=audit["complete_flight_observed"],
                  timing_passed=audit["calendar_timing_acceptance"], collisions=audit["counts"]["collision_events"]+
                  sum(v["final_snapshot_new_collisions"] for v in audit["vehicles"].values()))
    return result, peers


def compare(off, on, config):
    period = config["period_ms"]
    target = config["command_retime_probe"]
    start = target["cycle"]*period + config["initial_phase_ms"][target["vehicle"]]
    # The same releases in both experiments from injection to shortly after
    # the original 160 ms guard could fire. Full-run comparisons are separate.
    result = {}
    for label in ("first_220_ms", "full_run"):
        common = {k for k in off.keys() & on.keys() if label=="full_run" or
                  start <= k[1]*period+config["initial_phase_ms"][k[0]] < start+220}
        result[label] = dict(matched=len(common), off_misses=sum(off[k]!="delivered" for k in common),
                             on_misses=sum(on[k]!="delivered" for k in common),
                             newly_missed=sorted(k for k in common if off[k]=="delivered" and on[k]!="delivered"),
                             newly_delivered=sorted(k for k in common if off[k]!="delivered" and on[k]=="delivered"))
    result["unmatched_full_run"] = dict(only_off=len(off.keys()-on.keys()), only_on=len(on.keys()-off.keys()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "inputs", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for line in (args.runs / "inputs.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        path = args.inputs / name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    audit = {Path(r["data"]).parent.name:r for r in load(args.audit)["runs"]}
    result = {}
    for target in ("drone2", "drone5"):
        suffixes = (f"{target}_off", f"{target}_on")
        off, on = [load(args.inputs / f"config_{s}.json") for s in suffixes]
        assert off["command_retime_probe"]["retime"] is False and on["command_retime_probe"]["retime"] is True
        matching = dict(on, command_retime_probe=dict(on["command_retime_probe"], retime=False))
        assert off == matching
        case = load(args.inputs / "case.json")
        runs = [local_run(args.runs / s / "trial", cfg, case, audit[s]) for s,cfg in zip(suffixes,(off,on))]
        result[target] = dict(off=runs[0][0], on=runs[1][0], peers=compare(runs[0][1], runs[1][1], off))
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
