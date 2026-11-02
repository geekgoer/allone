"""Append only previously absent cells visited in all four recovery protocols."""

from pathlib import Path
import shutil

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_runtime import context_grid_key


BASE = ROOT/"controller_architecture_study/results"
SOURCE = BASE/"unknown_mode_recovery_20261026_v2"
ORIGINAL = BASE/"context_mode_map_v7/mode_map.json"
OUTPUT = BASE/"context_mode_map_v8"


def main():
    data,summary,plan = read_json(ORIGINAL),read_json(SOURCE/"summary.json"),read_json(SOURCE/"plan.json")
    if not summary["complete"] or len(summary["rows"])!=4 or not summary["target_found"]:
        raise ValueError("all four predeclared recovery conditions required")
    if plan["mode_map_sha256"]!=sha256(ORIGINAL) or plan["provenance"]!=data["provenance"]:
        raise ValueError("recovery source/map identity differs")
    target = read_json(SOURCE/"target.json")
    models,sets,counts,hashes = set(),[],{},{}
    for row in summary["rows"]:
        r = row["report"]
        if (not row["observable_initial_match"] or not r["control_recovery_passed"] or
                not r["sampled_update_bound_passed"] or r["max_actual_input_age_s"]>.030001):
            raise ValueError("observable match, recovery, update and measured-input-age checks required")
        directory = SOURCE/f"{row['model']}_{row['profile']}"
        for name in ("report.json","traces.json","commands.json"):
            hashes[str(directory/name)] = sha256(directory/name)
        keys = set()
        for state in read_json(directory/"traces.json")["Drone1"]:
            lo,hi = data["reference_acceleration_envelope"]["phase_range_s"]
            if not lo<=target["phase_s"]+state["t"]<=hi:
                continue
            key = context_grid_key(state["risk"],data["context_grid_steps"])
            keys.add(key)
            counts[key] = counts.get(key,0)+1
        sets.append(keys)
        models.add(row["model"])
    common = set.intersection(*sets)
    added = sorted(k for k in common if k not in data["context_grid"])
    # An existing failure, wait cap, or positive label is never overwritten.
    for key in added:
        data["context_grid"][key] = {"observations":counts[key],"source_conditions":4,
            "evidence_kind":"matched_observable_recovery_protocols_not_a_region_certificate",
            "modes":{m:{"status":"empirical_admissible","delay_limit_s":.040,
                "future_update_limit_s":.040,"input_age_limit_s":.030} for m in sorted(models)}}
    target_key = context_grid_key(target["state"]["risk"],data["context_grid_steps"])
    if target_key not in added:
        raise ValueError("captured unknown target must be a newly tested common cell")
    OUTPUT.mkdir(exist_ok=False)
    data.update(base_mode_map_sha256=sha256(ORIGINAL),base_builder_sha256=data["builder_sha256"],
        builder_sha256=sha256(__file__),independent_confirmation=False,
        extension={"source":str(SOURCE),"source_sha256":hashes,"added_grid_keys":added,
            "target_key":target_key,"all_four_protocols_visit_each_added_cell":True,
            "existing_grid_cells_changed":0,"runtime_input_age_cap_s":.030,
            "measured_max_input_age_s":max(r["report"]["max_actual_input_age_s"] for r in summary["rows"]),
            "calibration_timestamp_tolerance_s":.000001,
            "scope":"finite observed starts/suffixes only; not every state or switching sequence in a quantized cell; hidden actuator internal state reset"})
    data["compilation"].update(unknown_target_recovery_conditions=4,added_common_grid_cells=len(added))
    atomic_json(OUTPUT/"mode_map.json",data)
    shutil.copy2(Path(__file__),OUTPUT/"frozen_builder.py")
    print(f"CONTEXT_EXTENSION new_cells={len(added)} existing_cells_changed=0 target={target_key}")


if __name__=="__main__":
    main()
