"""Compile a SMALL empirical development map; quantization is not a safety proof."""

from pathlib import Path
import json

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .fast_convex_preprocess import FastReferenceWindow
from .fly_decision_dnn import StoredRoutePlanner
from .framework_runtime import CONTRACT, COLLISION_CONTRACT, context_grid_key, state_bin, H_EDGES, HDOT_EDGES
from .probe_airsim_three import load_layout
from .state_mode_calibration import moderate_case


RESULTS = ROOT/"controller_architecture_study/results"
SOURCE = RESULTS/"framework_runtime_v27_stable_static/n7_net2"
OUTPUT = RESULTS/"context_mode_map_v7"
STEPS = {"h":.025,"hdot":.100,"turn_pressure":.250}
MODEL = "plain_mlp_h9_w1024_seed0"


def reference_acceleration_bound(window, begin, end):
    maximum = 0.
    for c,start,duration in zip(window.coefficients,window.starts,window.durations):
        lo,hi = max(0.,begin-start),min(duration,end-start)
        if lo>hi:
            continue
        acceleration = c[:2,2:]*[2.,6.,12.,20.]
        components = []
        for a in acceleration:
            roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(a))
            times = [lo,hi,*[z.real for z in roots if abs(z.imag)<1e-8 and lo<=z.real<=hi]]
            components.append(max(abs(np.polynomial.polynomial.polyval(t,a)) for t in times))
        maximum = max(maximum,float(np.linalg.norm(components)))
    return float(np.nextafter(maximum,float("inf")))


def main():
    setup = read_json(SOURCE/"setup.json")
    if setup["collision_observation"]!=COLLISION_CONTRACT:
        raise ValueError("verified collision source required")
    _,layout = moderate_case()
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory,planner.config)
    grid,sources,conflicts = {},{},[]

    def features(row, phase, reference_window=window):
        velocity = np.array(row["reference_velocity"])
        normal = np.array([-velocity[1],velocity[0],0.])
        normal /= max(float(np.linalg.norm(normal)),1e-12)
        acceleration = reference_window._evaluate(np.array([phase+row["t"]]))[2][0]
        error = np.array(row["position"])-row["reference_position"]
        return {**row["risk"],"turn_pressure":float(np.sign(error@normal)*(acceleration@normal))}

    def add_trace(directory, traces, phases, cap, reference_window=window):
        sources[str(directory/"report.json")] = sha256(directory/"report.json")
        sources[str(directory/"traces.json")] = sha256(directory/"traces.json")
        for vehicle,rows in traces.items():
            for row in rows:
                risk = features(row,phases[vehicle],reference_window)
                key = context_grid_key(risk,STEPS)
                cell = grid.setdefault(key,{"modes":{},"observations":0})
                cell["observations"] += 1
                config = cell["modes"].setdefault(MODEL,{"status":"empirical_admissible",
                    "delay_limit_s":cap,"future_update_limit_s":cap,"input_age_limit_s":.030,"successful_caps_s":[]})
                if cap not in config["successful_caps_s"]:
                    config["successful_caps_s"].append(cap)
                config["delay_limit_s"] = max(config["successful_caps_s"])
                config["future_update_limit_s"] = max(config["successful_caps_s"])

    nominal = [RESULTS/f"framework_runtime_v26_fine_static/n7_net2/calibration/{MODEL}_T{t}" for t in (40,45,50)]
    nominal += [SOURCE/f"baseline_validation/c0_r{i}" for i in (0,1)]
    phases = {f"Drone{i+1}":setup["phase_s"]-setup["physical_phase_span_s"]*i/6 for i in range(7)}
    for directory in nominal:
        report = read_json(directory/"report.json")
        if not report["accepted"] or set(report["configuration"]["models"].values())!={MODEL}:
            raise ValueError("only completed, matched constant-controller nominal trajectories may seed the map")
        add_trace(directory,read_json(directory/"traces.json"),phases,.060)

    calibrated = RESULTS/"framework_gate_task_calibration_v2_direct_collision"
    calibration_plan = read_json(calibrated/"plan.json")
    if (calibration_plan["collision_observation"]!=COLLISION_CONTRACT
            or calibration_plan["provenance"]!=setup["provenance"]):
        raise ValueError("matching authoritative-collision calibration required")
    straight,_ = load_layout(48,0)
    straight_planner = StoredRoutePlanner(straight)
    windows = {"turn":window,"straight":FastReferenceWindow(straight_planner.trajectory,straight_planner.config)}
    calibrated_count,pressure_by_bin = 0,{}
    for path in sorted(calibrated.glob(f"*_{MODEL}/report.json")):
        report = read_json(path)
        if not report["recovery_passed"] or not report["timing_comparable"]:
            continue
        geometry = path.parent.name.split("_")[0]
        phase = calibration_plan["geometry"][geometry]["phase_s"]
        trace = read_json(path.parent/"traces.json")
        add_trace(path.parent,{"Drone1":trace},{"Drone1":phase},.080,windows[geometry])
        for row in trace:
            risk = features(row,phase,windows[geometry])
            pressure_by_bin.setdefault(state_bin(risk),[]).append(risk["turn_pressure"])
        calibrated_count += 1

    recurring_root = RESULTS/"recurring_input_age_v2"
    recurring = read_json(recurring_root/"summary.json")
    cases = read_json(RESULTS/"wait_sensitive_states_refinement_v1/plan.json")["cases"]
    if not recurring["complete"] or read_json(recurring_root/"plan.json")["provenance"]!=setup["provenance"]:
        raise ValueError("completed recurrent-age data with matching controllers required")
    used = 0
    for row in recurring["rows"]:
        if not row["report"]["control_recovery_passed"] or not row["report"]["sampled_update_bound_passed"]:
            continue
        case = next(c for c in cases if c["id"]==row["source_case"])
        phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/6
        directory = Path(row["directory"])
        add_trace(directory,read_json(directory/"traces.json"),{"Drone1":phase},.040)
        used += 1

    # Immediate-control failures veto their quantized cells. Later-wait failures
    # must NOT veto a state that has a successful shorter wait protocol.
    negatives = []
    for row in read_json(RESULTS/"mode_context_capacity_v1/summary.json")["rows"]:
        if row["model"]==MODEL and not row["report"]["control_recovery_passed"]:
            phase = read_json(RESULTS/"mode_projection_probe_v1/plan.json")["phases"][row["context"]]
            negatives.append((Path(row["directory"]),phase))
    for name in ("initial_recovery_window_v1","wait_sensitive_states_v1"):
        for row in read_json(RESULTS/name/"summary.json")["rows"]:
            r = row.get("report",row)
            if r.get("controller","DNN")!="DNN" or r.get("requested_initial_wait_s")!=0. or r["control_recovery_passed"]:
                continue
            case = next(c for c in cases if c["id"]==r.get("source_case",row.get("source_case")))
            phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/6
            directory = Path(row["directory"]) if "directory" in row else RESULTS/name/f"{case['id']}_DNN_delay0"
            negatives.append((directory,phase))
    for directory,phase in negatives:
        row = read_json(directory/"traces.json")["Drone1"][0]
        key = context_grid_key(features(row,phase),STEPS)
        if key in grid:
            conflicts.append(key)
        grid[key] = {"modes":{},"status":"observed_failure","negative_report_sha256":sha256(directory/"report.json")}

    # A shorter successful protocol is not evidence that a longer one failed.
    # Retain the largest observed successful cap, but NEVER cross a known
    # failing wait for an initial state in the same quantized cell.
    for name in ("wait_sensitive_states_v1","wait_sensitive_states_refinement_v1"):
        for item in read_json(RESULTS/name/"summary.json")["rows"]:
            if item["report"]["control_recovery_passed"]:
                continue
            case = next(c for c in cases if c["id"]==item["source_case"])
            phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/6
            directory = Path(item["directory"])
            row = read_json(directory/"traces.json")["Drone1"][0]
            key = context_grid_key(features(row,phase),STEPS)
            cell = grid.setdefault(key,{"modes":{}})
            cell["first_failed_wait_s"] = min(cell.get("first_failed_wait_s",float("inf")),item["delay_s"])
    for cell in grid.values():
        for model,config in list(cell["modes"].items()):
            permitted = [cap for cap in config["successful_caps_s"] if cap<cell.get("first_failed_wait_s",float("inf"))]
            if not permitted:
                del cell["modes"][model]
            else:
                config["delay_limit_s"] = config["future_update_limit_s"] = max(permitted)

    base_path = RESULTS/"framework_runtime_refinement_v2_direct_collision/mode_map.json"
    base = read_json(base_path)
    if base["provenance"]!=setup["provenance"] or base["collision_observation"]!=COLLISION_CONTRACT:
        raise ValueError("matching directly calibrated fallback required")
    fallback = []
    phase_range = [setup["phase_s"]-setup["physical_phase_span_s"],setup["phase_s"]+setup["duration_s"]]
    pressure_bound = reference_acceleration_bound(window,*phase_range)
    for key,cell in base["bins"].items():
        modes = {m:{**s,"future_update_limit_s":.080,"input_age_limit_s":.030}
            for m,s in cell["modes"].items() if s.get("status")=="empirical_admissible"}
        if not modes or key not in pressure_by_bin:
            continue
        hi,vi = (int(s[1:]) for s in key.split("_"))
        fallback.append({"id":"calibrated_"+key,"state_bin":key,"evidence_kind":"direct_recovery_calibration",
            "h_range":[H_EDGES[hi],H_EDGES[hi+1]],"hdot_range":[HDOT_EDGES[vi],HDOT_EDGES[vi+1]],
            "turn_pressure_range":[-pressure_bound,pressure_bound],"modes":modes,
            "context_scope_inherited_from_2d_calibration":True})
    for cell in base.get("local_cells",[]):
        modes = {m:{**s,"future_update_limit_s":.080,"input_age_limit_s":.030}
            for m,s in cell["modes"].items() if s.get("status")=="empirical_admissible"}
        if modes:
            fallback.insert(0,{**cell,"evidence_kind":"direct_recovery_calibration",
                "turn_pressure_range":[-pressure_bound,pressure_bound],"modes":modes,
                "context_scope_inherited_from_2d_calibration":True})
    result = {"contract":CONTRACT,"collision_observation":COLLISION_CONTRACT,"safety_predicate":"gate_task",
        "provenance":setup["provenance"],"net_width_m":2.,"complete":True,"bins":{},"local_cells":fallback,
        "context_feature":"turn_pressure","context_grid_steps":STEPS,"context_grid_alignment":"centered","context_grid":grid,
        "analysis_to_next_command_limit_s":.080,"source_reports_and_traces_sha256":sources,
        "compilation":{"nominal_episodes":len(nominal),"validated_80ms_calibration_episodes":calibrated_count,"direct_fallback_cells":len(fallback),"recurrent_age_episodes":used,"veto_points":len(negatives),
            "positive_negative_cell_conflicts":conflicts},"development_only":True,
        "scope":"quantized states from completed constant-controller episodes, with known-failure veto; NOT a verified region or arbitrary switching guarantee",
        "capacity_adaptation_enabled":True,"normal_update_cap_s":.080,"recovery_update_cap_s":.040,
        "merge_rule":"largest observed successful protocol cap below every known failing wait; shorter success alone is not a required deadline",
        "lookup_priority":"known failing waits veto first; direct recovery calibration precedes successful rollout suffixes",
        "input_age_cap_s":.030,"is_safety_certificate":False,"test_read":False,
        "direct_calibration_map_sha256":sha256(base_path),
        "fallback_scope":"retain the original sign-agnostic2D empirical regions within the observed reference-acceleration envelope; NOT per-context recovery certification",
        "reference_acceleration_envelope":{"phase_range_s":phase_range,"upper_bound":pressure_bound,
            "method":"norm of componentwise polynomial extrema at endpoints and derivative roots; conservative reference envelope, not interval-arithmetic certification",
            "route_sha256":setup["route_sha256"]},
        "builder_sha256":sha256(__file__)}
    OUTPUT.mkdir(exist_ok=False)
    atomic_json(OUTPUT/"mode_map.json",result)
    print(json.dumps({"output":str(OUTPUT/"mode_map.json"),"grid_cells":len(grid),**result["compilation"]},indent=2))


if __name__=="__main__":
    main()
