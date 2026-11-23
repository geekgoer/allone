"""Prepare and audit paired control-update omissions on the Zynq/AirSim rig."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .critical_update_screen import load, rows


PERIOD_MS = 20
MODEL_ID = 11
MARGIN_EFFECT_M = 0.02


@dataclass(frozen=True)
class Pilot:
    vehicle: str
    gate: int
    gate_time_s: float
    cycles: dict
    count: int
    vehicles: tuple


PILOTS = {
    "single": Pilot("Drone2", 8, 17.1154701202061, {"early": 729, "late": 774}, 1,
                    tuple(f"Drone{i}" for i in range(1, 13))),
    "burst3": Pilot("Drone3", 11, 24.6154701202061, {"early": 1114, "late": 1148}, 3,
                    ("Drone1", "Drone2", "Drone3", "Drone4")),
}


def prepare(case_path, config_path, output, pilot):
    case = load(case_path)
    config = load(config_path)
    gate = next(g for g in case["layouts"][pilot.vehicle]["gates"] if g["index"] == pilot.gate)
    if case["seed"] != 0 or abs(gate["reference_time_s"] - pilot.gate_time_s) > 1e-9:
        raise ValueError("Use the frozen twelve-vehicle case from the historical screen")
    if config["period_ms"] != PERIOD_MS or config["model_id"] != MODEL_ID or "initial_periods" in config:
        raise ValueError("Pilot requires the frozen uniform 20 ms and ID 11 starting configuration")
    output.mkdir(parents=True, exist_ok=False)
    for phase, cycle in pilot.cycles.items():
        for arm in ("normal", "omit"):
            run_config = dict(config, adaptive=False, enable_skip=False, critical_update_probe={
                "vehicle": pilot.vehicle, "cycle": cycle, "omit_update": arm == "omit"})
            if pilot.count != 1:
                run_config["critical_update_probe"]["count"] = pilot.count
            (output / f"{phase}_{arm}.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    (output / "case.json").write_bytes(case_path.read_bytes())
    (output / "vehicles.txt").write_bytes((",".join(pilot.vehicles) + "\n").encode("ascii"))
    files = sorted(output.iterdir())
    (output / "inputs.sha256").write_bytes("".join(
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files).encode("ascii"))


def profile(trial, analysis, phase, arm, pilot):
    cycle = pilot.cycles[phase]
    recovery_cycle = cycle + pilot.count
    identity = load(trial / "execution_identity.json")
    prepared = load(trial / "prepared_source.json")
    config = load(trial / "config_source.json")
    probe = config.pop("critical_update_probe")
    if probe.pop("count", 1) != pilot.count or probe != {
            "vehicle": pilot.vehicle, "cycle": cycle, "omit_update": arm == "omit"}:
        raise ValueError(f"Unexpected configured intervention in {trial}")
    if (identity["policy"], identity["period_ms"], identity["model_id"], identity["enable_skip"]) != (
            "fixed", PERIOD_MS, MODEL_ID, False):
        raise ValueError(f"Unexpected model or release calendar in {trial}")
    if sorted(prepared["vehicles"]) != sorted(pilot.vehicles):
        raise ValueError(f"Unexpected vehicles in {trial}")
    relevant = {}
    target_status = {}
    gate_objects = {part["actual_name"] for part in prepared["gate_objects"]
                    if part["vehicle"] == pilot.vehicle and part["gate_index"] == pilot.gate}
    requested = marked = 0
    contact = False
    peer_statuses = defaultdict(Counter)
    for row in rows(trial):
        marked += row.get("critical_update_probe") is True
        requested += row["status"] == "probe_omitted_update"
        if row["vehicle"] != pilot.vehicle:
            peer_statuses[row["vehicle"]][row["status"]] += 1
            continue
        target_status[row["cycle"]] = row["status"]
        if cycle - 2 <= row["cycle"] <= recovery_cycle + 1:
            relevant[row["cycle"]] = row
        contact |= row["cycle"] >= cycle and row.get("collision_object") in gate_objects
    if marked > pilot.count or requested > (pilot.count if arm == "omit" else 0):
        raise ValueError(f"Extra interventions in {trial}")
    target = relevant.get(cycle, {})
    expected_status = "probe_omitted_update" if arm == "omit" else "delivered"
    checked = analysis["vehicles"][pilot.vehicle]
    # Record final-snapshot contacts too, using the same named gate actors.
    contact |= any(hit.get("vehicle") == pilot.vehicle and hit.get("collision_new") is True and
                   hit.get("collision_object") in gate_objects for hit in load(trial / "final_states.json"))
    gate = next((g for g in checked["gate_crossings"]["gates"] if g["gate"] == pilot.gate), {})
    crossing = gate.get("first_crossing")
    sampled_after_start = (target.get("sample_start_us", math.inf) - identity["origin_us"]) / 1e6
    margin = (crossing["body_margin_m"] if crossing and crossing["telemetry_gap_acceptable"] and
              crossing["time_s"] >= sampled_after_start else None)
    faults = []
    if not analysis["audit_passed"] or analysis["runtime_failed"]:
        faults.append("lifecycle_or_runtime")
    probes = [relevant.get(k, {}) for k in range(cycle, recovery_cycle)]
    if any(r.get("status") != expected_status or not r.get("critical_update_probe") or not r.get("observe_accepted")
           for r in probes):
        faults.append("target_not_as_planned")
    if arm == "omit":
        if any(k in r for r in probes for k in ("dpu_enqueue_us", "command_submit_us", "new_held_version")):
            faults.append("omitted_task_did_work")
        if target.get("held_version") is None or any(
                r.get("held_version") != target["held_version"] for r in (*probes, relevant.get(recovery_cycle, {}))):
            faults.append("old_command_version_changed")
    if any(relevant.get(k, {}).get("status") != "delivered"
           for k in (cycle - 2, cycle - 1, recovery_cycle, recovery_cycle + 1)):
        faults.append("another_miss_next_to_target")
    ack_cycles = (cycle - 1, recovery_cycle) if arm == "omit" else range(cycle - 1, recovery_cycle + 1)
    acks = [relevant.get(k, {}).get("command_return_us") for k in ack_cycles]
    gap_ms = max(b - a for a, b in zip(acks, acks[1:])) / 1000 if None not in acks else None
    if margin is None and not contact:
        faults.append("no_reliable_gate_outcome")
    after_cycle = crossing.get("after_cycle") if crossing else None
    extra_misses = (dict(Counter(target_status.get(k, "missing_record")
                                for k in range(recovery_cycle, after_cycle + 1)
                                if target_status.get(k) != "delivered"))
                    if isinstance(after_cycle, int) else None)
    if pilot.count > 1 and extra_misses:
        faults.append("another_miss_before_crossing")
    return config, prepared["original_case"], target, dict(faults=faults, contact=contact, margin_m=margin,
                                                           peer_statuses=peer_statuses,
                                                           extra_misses_until_crossing=extra_misses,
                                                           longest_local_ack_gap_ms=gap_ms)


def compare(root, audit, pilot):
    runs = {Path(r["data"]).parent.name: r for r in audit["runs"]}
    result = {}
    for phase in pilot.cycles:
        pairs = []
        for repeat in (1, 2):
            settings, cases, samples, metrics = [], [], [], []
            for arm in ("normal", "omit"):
                name = f"{phase}_{repeat}_{arm}"
                cfg, case, sample, record = profile(root / name / "trial", runs[name], phase, arm, pilot)
                settings.append(cfg)
                cases.append(case)
                samples.append(sample)
                metrics.append(record)
            if settings[0] != settings[1] or cases[0] != cases[1]:
                raise ValueError("Pair did not use the same case and costs")
            a, b = samples
            differences = {}
            for key, bound in (("position", .04), ("velocity", .15), ("angular_velocity", .15),
                               ("quaternion", .025), ("held_control", .30)):
                distance = math.inf
                if key in a and key in b:
                    distance = math.dist(a[key], b[key])
                    if key == "quaternion":
                        distance = min(distance, math.dist(a[key], [-x for x in b[key]]))
                differences[key] = dict(distance=distance if math.isfinite(distance) else None, limit=bound,
                                        within_limit=distance <= bound)
            if all("sample_start_us" in s and "planned_release_us" in s for s in samples):
                sample_lag_ms = abs((a["sample_start_us"] - a["planned_release_us"]) -
                                    (b["sample_start_us"] - b["planned_release_us"])) / 1000
            else:
                sample_lag_ms = None
            differences["sample_lag_ms"] = dict(distance=sample_lag_ms, limit=5,
                                                within_limit=sample_lag_ms is not None and sample_lag_ms <= 5)
            normal, omit = metrics
            valid = all(d["within_limit"] for d in differences.values()) and not (normal["faults"] or omit["faults"])
            loss = normal["margin_m"] - omit["margin_m"] if None not in (normal["margin_m"], omit["margin_m"]) else None
            pairs.append(dict(repeat=repeat, valid=valid, start_differences=differences, normal=normal, omit=omit,
                              margin_loss_m=loss, meaningful_loss=valid and not normal["contact"] and
                              (normal["margin_m"] or 0) > 0 and (
                                  omit["contact"] or (loss is not None and loss >= MARGIN_EFFECT_M))))
        result[phase] = dict(pairs=pairs, repeats_with_effect=sum(p["meaningful_loss"] for p in pairs),
                             nomination_for_confirmation=all(p["meaningful_loss"] for p in pairs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("prepare")
    for name in ("case", "config", "output"):
        generate.add_argument(f"--{name}", type=Path, required=True)
    evaluate = commands.add_parser("compare")
    for name in ("root", "audit", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    for command in (generate, evaluate):
        command.add_argument("--pilot", choices=PILOTS, default="single")
    args = parser.parse_args()
    pilot = PILOTS[args.pilot]
    if args.command == "prepare":
        prepare(args.case, args.config, args.output, pilot)
    else:
        result = compare(args.root, load(args.audit), pilot)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print({phase: part["repeats_with_effect"] for phase, part in result.items()})


if __name__ == "__main__":
    main()
