"""Causal shadow replay of the PPT gate-clearance budget, without control writes.

At sample k: T_max = min(d_y / v_y_bound, d_z / v_z_bound) - R_next_estimate.
Afterwards compare the actual sample-k to command-(k+1) age, not R_k.
No sampled velocity or response quantile is promoted to a guaranteed bound.
"""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from environment import Gate
from gate_route import _gate_local
from .decision_dataset import ROOT, atomic_json, sha256


DEFAULT_CONFIG = ROOT / "controller_architecture_study/configs/state_budget_shadow_v1.json"


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    positive = ("monitor_half_depth_m", "lateral_error_rate_bound_mps",
                "vertical_error_rate_bound_mps", "response_floor_ms")
    if not all(np.isfinite(config[k]) and config[k] > 0 for k in positive):
        raise ValueError("budget distances, rate bounds and response floor must be positive and finite")
    if not np.isfinite(config["response_margin_ms"]) or config["response_margin_ms"] < 0:
        raise ValueError("response margin must be finite and nonnegative")
    history = config["response_history_size"]
    if isinstance(history, bool) or not isinstance(history, int) or history <= 0:
        raise ValueError("response history size must be a positive integer")
    periods = config["candidate_periods_ms"]
    if not periods or not all(np.isfinite(t) and t > 0 for t in periods):
        raise ValueError("candidate periods must be positive and finite")
    return config


def gate_budget(position, velocity, gates, geometry, config):
    """Static gate-local axes; keep negative clearance and an explicit domain."""
    limits = np.array([geometry["width"], geometry["height"]])/2
    limits -= geometry["thickness"]/2 + geometry["clearance"]
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("invalid clear gate opening")
    bounds = np.array([config["lateral_error_rate_bound_mps"], config["vertical_error_rate_bound_mps"]])
    monitored = []
    for gate in gates:
        local = _gate_local(np.asarray(position), gate)
        if abs(local[0]) > config["monitor_half_depth_m"]:
            continue
        local_velocity = _gate_local(gate.center + np.asarray(velocity), gate)
        clearance = limits - np.abs(local[1:])
        times = 1000*clearance/bounds
        axis = int(np.argmin(times))
        monitored.append({"gate": gate.index, "local_position_m": local.tolist(),
            "local_velocity_mps": local_velocity.tolist(), "remaining_clearance_m": clearance.tolist(),
            "validity_budget_ms": float(times[axis]), "limiting_axis": ("lateral", "vertical")[axis],
            "observed_rate_bound_exceeded": bool(np.any(np.abs(local_velocity[1:]) > bounds)),
            "outside_inner_opening": bool(np.any(clearance < 0))})
    return {"applicable": bool(monitored), "constraints": monitored,
            "validity_budget_ms": min((c["validity_budget_ms"] for c in monitored), default=None),
            "observed_rate_bound_exceeded": any(c["observed_rate_bound_exceeded"] for c in monitored),
            "outside_inner_opening": any(c["outside_inner_opening"] for c in monitored)}


def statistics(values):
    return None if not values else {"min": float(min(values)), "mean": float(np.mean(values)),
                                  "p95": float(np.percentile(values, 95)), "max": float(max(values))}


def analyze(traces, requests, local_gates, geometry, config):
    """Use only responses dispatched before sample k for its R_next estimate."""
    lookup, events = {}, []
    for request in requests:
        if "sample_start_s" not in request:
            continue  # A skipped release has no physical state sample.
        key = request["vehicle"], request["sample_start_s"]
        if key in lookup:
            raise ValueError("duplicate vehicle/sample timestamp")
        lookup[key] = request
        if "dnn_dispatch_s" in request:
            response = 1000*(request["dnn_dispatch_s"] - request["sample_start_s"])
            if not np.isfinite(response) or response < 0:
                raise ValueError("invalid dispatch response time")
            events.append((request["dnn_dispatch_s"], response))
    events.sort()
    states, next_requests = [], {}
    for vehicle, trace in traces.items():
        if vehicle not in local_gates:
            raise ValueError("missing gate layout for a vehicle")
        gates = [Gate(g["index"], g["center"], g["orientation"]) for g in local_gates[vehicle]]
        for index, sample in enumerate(trace):
            request = lookup[(vehicle, sample["sample_s"])]
            if index and sample["sample_s"] <= trace[index-1]["sample_s"]:
                raise ValueError("flight samples must be strictly ordered per vehicle")
            if not np.isfinite([*sample["position"], *sample["velocity"], sample["sample_s"], request["period_ms"]]).all():
                raise ValueError("nonfinite flight sample")
            state = {"vehicle": vehicle, "cycle": request["cycle"], "sample_s": sample["sample_s"],
                     "model": request["model"], "period_ms": request["period_ms"],
                     **gate_budget(sample["position"], sample["velocity"], gates, geometry, config)}
            states.append(state)
            if index+1 < len(trace):
                next_requests[(vehicle, request["cycle"])] = lookup[(vehicle, trace[index+1]["sample_s"])]
    states.sort(key=lambda r: r["sample_s"])
    history = deque(maxlen=config["response_history_size"])
    event_index = 0
    for state in states:
        # Strictly prior dispatches, across all three clients sharing the worker.
        while event_index < len(events) and events[event_index][0] < state["sample_s"]:
            history.append(events[event_index][1])
            event_index += 1
        reserve = max(config["response_floor_ms"], max(history, default=0.) + config["response_margin_ms"])
        budget = state["validity_budget_ms"]
        cap = None if budget is None else budget-reserve
        usable = cap is not None and not state["observed_rate_bound_exceeded"]
        state.update(next_response_estimate_ms=reserve, response_history_count=len(history),
            T_max_estimate_ms=cap, estimate_usable_at_sample=usable,
            nominal_period_within_estimated_budget=None if not usable else state["period_ms"] <= cap,
            feasible_periods_same_response_ms=[] if not usable else [t for t in config["candidate_periods_ms"] if t <= cap],
            actual_next_sample_interval_ms=None, actual_next_response_ms=None,
            actual_age_at_next_dispatch_ms=None, posthoc_budget_exceeded=None,
            next_response_exceeded_estimate=None, next_cycle_contiguous=None)
        following = next_requests.get((state["vehicle"], state["cycle"]))
        if following is not None and "dnn_dispatch_s" in following:
            interval = 1000*(following["sample_start_s"]-state["sample_s"])
            response = 1000*(following["dnn_dispatch_s"]-following["sample_start_s"])
            age = interval+response
            state.update(actual_next_sample_interval_ms=interval, actual_next_response_ms=response,
                actual_age_at_next_dispatch_ms=age,
                posthoc_budget_exceeded=None if budget is None else age > budget,
                next_response_exceeded_estimate=response > reserve,
                next_cycle_contiguous=following["cycle"] == state["cycle"]+1)
    active = [r for r in states if r["applicable"]]
    compared = [r for r in active if r["posthoc_budget_exceeded"] is not None]
    summary = {"mode": "causal_shadow_replay", "control_changed": False, "is_safety_certificate": False,
        "formula": "T_k + R_(k+1) <= min(d_y/v_y_bound, d_z/v_z_bound)",
        "geometry": geometry, "config": config,
        "samples": len(states), "applicable_samples": len(active),
        "not_applicable_samples": len(states)-len(active),
        "sampled_requests_without_flight_trace": len(lookup)-len(states),
        "observed_rate_bound_exceedances": sum(r["observed_rate_bound_exceeded"] for r in active),
        "outside_inner_opening_samples": sum(r["outside_inner_opening"] for r in active),
        "nominal_period_over_budget_samples": sum(r["nominal_period_within_estimated_budget"] is False for r in active),
        "posthoc_comparisons": len(compared),
        "posthoc_over_budget_samples": sum(r["posthoc_budget_exceeded"] for r in compared),
        "applicable_without_next_dispatch": len(active)-len(compared),
        "next_response_estimate_exceedances": sum(r["next_response_exceeded_estimate"] for r in compared),
        "noncontiguous_next_cycles": sum(r["next_cycle_contiguous"] is False for r in states),
        "T_max_estimate_ms": statistics([r["T_max_estimate_ms"] for r in active]),
        "age_at_next_dispatch_ms": statistics([r["actual_age_at_next_dispatch_ms"] for r in compared]),
        "candidate_period_feasible_samples": {str(t): sum(t in r["feasible_periods_same_response_ms"] for r in active)
                                               for t in config["candidate_periods_ms"]},
        "limits": ["No model-quality or MPC-error bound is included.",
                   "Same-response candidate counts are not model-specific feasibility predictions.",
                   "No applicable samples means no evidence, not a pass.",
                   "A violated bound invalidates the conditional estimate, not proof of an actual collision.",
                   "Measurements do not observe true sensor acquisition or actuator application times."]}
    return summary, states


def save_analysis(path, traces, requests, local_gates, geometry, config):
    summary, states = analyze(traces, requests, local_gates, geometry, config)
    atomic_json(path / "state_budget_samples.json", states)
    atomic_json(path / "state_budget_report.json", summary)
    return summary


def restore_local_gates(report, traces):
    """Old reports lack local offsets: recover the constant shift from references."""
    if "vehicle_local_gates" in report:
        return report["vehicle_local_gates"], "recorded_vehicle_local_gates"
    from .probe_airsim_three import load_layout
    from .fly_decision_dnn import StoredRoutePlanner
    from .fast_convex_preprocess import FastReferenceWindow

    layout, route_hash = load_layout(report["environment_id"], report["route_index"])
    if route_hash != report["saved_route_sha256"]:
        raise ValueError("saved route changed")
    planner = StoredRoutePlanner(layout)
    window = FastReferenceWindow(planner.trajectory, planner.config)
    result = {}
    for vehicle, trace in traces.items():
        if not trace:
            raise ValueError("cannot recover coordinates without flight samples")
        shifts = [np.asarray(r["reference_position"])-window.window(r["reference_elapsed_s"])[0].position for r in trace]
        shift = shifts[0]
        if not np.allclose(shifts, shift, rtol=0., atol=1e-6):
            raise ValueError("reference translation was not constant")
        result[vehicle] = [{**g, "center": (np.asarray(g["center"])+shift).tolist()} for g in layout["gates"]]
    return result, "recovered_from_hash_verified_route_and_constant_reference_translation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="New directory; never overwrite the flight report")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = [args.report, args.report.parent / "requests.jsonl", args.report.parent / "flight_trace.json"]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["mode"] != "dnn":
        raise ValueError("this replay requires actual DNN command-dispatch timestamps, not MPC-controlled shadow flight")
    requests = [json.loads(line) for line in sources[1].read_text(encoding="utf-8").splitlines()]
    traces = json.loads(sources[2].read_text(encoding="utf-8"))
    gates, frame_source = restore_local_gates(report, traces)
    from .probe_airsim_three import GATE
    from .dynamic_probe_common import new_output

    path = args.output_dir
    if path is None:
        path = new_output("state_budget_shadow")
    else:
        path.mkdir(parents=True, exist_ok=False)
    summary = save_analysis(path, traces, requests, gates, report.get("gate_geometry", GATE), config)
    if hashes != {str(p.resolve()): sha256(p) for p in sources}:
        raise RuntimeError("source files changed during replay")
    summary.update(source_sha256=hashes, config_sha256=sha256(args.config), gate_frame_source=frame_source,
                   observer_source_sha256=sha256(Path(__file__)),
                   original_report_rewritten=False, airsim_started=False, gpu_inference_started=False)
    atomic_json(path / "state_budget_report.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "source_sha256", "limits")}, indent=2))
    print(f"Report: {path / 'state_budget_report.json'}")


if __name__ == "__main__":
    main()
