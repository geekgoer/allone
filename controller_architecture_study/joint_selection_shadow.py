"""Post-flight causal joint-selection shadow on an unchanged DNN flight.

All three policies see the same observed baseline queue and states. Their
recommendations are NOT applied or simulated as counterfactual flight outcomes.
"""

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import threading
import time

import numpy as np

from environment import Gate
from .decision_dataset import atomic_json, sha256
from .dynamic_probe_common import describe, new_output, read_json
from .joint_controller_sweep import RUNTIME_KEYS
from .online_joint_policy import DEFAULT_SUMMARY
from .state_budget_shadow import DEFAULT_CONFIG, gate_budget, load_config, restore_local_gates
from .state_joint_policy import StateResourceSelector, load_joint_profile


MODES = ("fixed", "period_only", "adaptive")
DISPATCH_FIELDS = ("sample_start_s", "enqueue_s", "ready_s", "dnn_dispatch_s")


class JointShadowMonitor:
    def __init__(self, profile, config, vehicles):
        self.profile, self.config, self.vehicles = profile, config, tuple(vehicles)
        self.policies = {v: {m: StateResourceSelector(profile, config) for m in MODES} for v in vehicles}
        self.lock = threading.Lock()
        self.dispatches = deque(maxlen=4096)
        self.decisions = []
        self.alignment = None

    def record_dispatch(self, row):
        record = {k: row[k] for k in DISPATCH_FIELDS}
        with self.lock:
            self.dispatches.append(record)

    def observe(self, vehicle, cycle, sample_s, scheduled_s, now, state, snapshot, actual_choice):
        started = time.perf_counter()
        with self.lock:
            history = tuple(self.dispatches)
        choices = {mode: self.policies[vehicle][mode].select(len(self.vehicles), now, snapshot, state,
            mode=mode, producer_elapsed_ms=1000*(now-sample_s),
            release_elapsed_ms=1000*(now-scheduled_s), dispatch_history=history) for mode in MODES}
        record = {"vehicle": vehicle, "cycle": cycle, "sample_s": sample_s, "scheduled_s": scheduled_s, "decision_s": now,
            "actual_model": actual_choice["model"], "actual_period_ms": actual_choice["period_ms"],
            "state": state, "policies": choices,
            "decision_ms": 1000*(time.perf_counter()-started), "recommendations_applied": False}
        with self.lock:
            self.decisions.append(record)
        return record

    def save(self, path, *, execution_mode, source_status):
        decisions = sorted(self.decisions, key=lambda r: (r["decision_s"], r["vehicle"]))
        policies = {}
        for mode in MODES:
            rows = [r["policies"][mode] for r in decisions]
            recommendations = [r["recommendation"] for r in rows if r["recommendation"] is not None]
            policies[mode] = {"decisions": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
                "recommendations": len(recommendations),
                "model_counts": dict(Counter(r["model"] for r in recommendations)),
                "period_counts_ms": dict(Counter(str(r["period_ms"]) for r in recommendations)),
                "recommendation_changes": sum(r["changed"] for r in rows)}
        summary = {"status": "COMPLETE" if source_status == "COMPLETE" else "SOURCE_INCOMPLETE",
            "execution_mode": execution_mode, "source_status": source_status, "samples": len(decisions),
            "applicable_samples": sum(r["state"]["applicable"] for r in decisions),
            "control_changed": False, "recommendations_applied": False, "is_safety_certificate": False,
            "request_alignment": self.alignment,
            "policies": policies, "decision_ms": describe([r["decision_ms"] for r in decisions]),
            "decision_timing_scope": "Offline analysis CPU time; not part of the recorded flight response.",
            "profile_summary_sha256": self.profile["summary_sha256"], "budget_config": self.config,
            "quality_scope": self.profile["quality_scope"],
            "limits": ["Recommendations use the observed baseline queue, not a resimulated policy workload.",
                "No inference/control/trajectory benefit is measured for a hypothetical recommendation.",
                "Resource load uses a homogeneous N*C/T approximation, not heterogeneous schedulability analysis.",
                "P95/prior estimates and gate-rate assumptions are not certified upper bounds.",
                "Outside gate neighborhoods only resource feasibility is estimated.",
                "The response prediction includes producer, queue, inference and async dispatch, not actuator acknowledgement."]}
        with (path / "joint_shadow_decisions.jsonl").open("w", encoding="utf-8") as handle:
            for record in decisions:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
        atomic_json(path / "joint_shadow_report.json", summary)
        return summary


def replay(profile, config, report, requests, traces, local_gates, geometry):
    """Deliver enqueue/ready/dispatch events only when their timestamps have passed."""
    if report["mode"] != "dnn":
        raise ValueError("joint shadow replay requires actual DNN dispatches")
    if any(report["runtime"].get(k) != profile["runtime"].get(k) for k in RUNTIME_KEYS):
        raise ValueError("source flight runtime differs from the measured candidate profile")
    if report["provenance"] != profile["provenance"]:
        raise ValueError("source flight uses different model/data provenance")
    lookup, events = {}, []
    for row in requests:
        if "sample_start_s" not in row:
            continue
        key = row["vehicle"], row["sample_start_s"]
        if key in lookup:
            raise ValueError("duplicate source sample")
        lookup[key] = row
        if row.get("status") != "ready":
            continue
        fields = ["sample_start_s", "feature_ready_s", "enqueue_s", "start_s", "ready_s"]
        if "dnn_dispatch_s" in row:
            fields.append("dnn_dispatch_s")
        chain = [row[k] for k in fields]
        if not np.isfinite(chain).all() or np.any(np.diff(chain) < 0):
            raise ValueError("invalid source timing chain")
        for kind, field in enumerate(("enqueue_s", "ready_s", "dnn_dispatch_s")):
            if field not in row:
                continue
            events.append((row[field], kind, row))
    events.sort(key=lambda e: (e[0], e[1]))
    gates = {v: [Gate(g["index"], g["center"], g["orientation"]) for g in layout] for v, layout in local_gates.items()}
    legacy = {(v, r["sample_s"]): r for v, trace in traces.items() for r in trace}
    samples = []
    for key, row in lookup.items():
        sample = row.get("joint_shadow_state", legacy.get(key))
        if sample is None:
            continue
        now = row.get("joint_shadow_snapshot_s", row["feature_ready_s"])
        if not np.isfinite(now) or not row["feature_ready_s"] <= now <= row["enqueue_s"]:
            raise ValueError("invalid source capture timing")
        samples.append((now, row["vehicle"], sample, row))
    samples.sort(key=lambda item: (item[0], item[1]))
    monitor = JointShadowMonitor(profile, config, traces)
    event_index, pending = 0, {}
    recent = deque(maxlen=4096)
    for now, vehicle, sample, row in samples:
        while event_index < len(events) and events[event_index][0] < now:
            _, kind, event_row = events[event_index]
            key = event_row["vehicle"], event_row["cycle"]
            if kind == 0:
                pending[key] = event_row["model"]  # No future service time/output is inspected here.
            elif kind == 1:
                pending.pop(key, None)
                recent.append({k: event_row[k] for k in ("model", "status", "ready_s", "service_ms", "sample_start_s", "scheduled_s")})
            else:
                monitor.record_dispatch(event_row)
            event_index += 1
        while recent and recent[0]["ready_s"] < now-2.:
            recent.popleft()
        snapshot = {"recent": list(recent), "pending_model_counts": dict(Counter(pending.values()))}
        state = gate_budget(sample["position"], sample["velocity"], gates[vehicle], geometry, config)
        monitor.observe(vehicle, row["cycle"], row["sample_start_s"], row["scheduled_s"], now, state, snapshot, row)
    matched = 0
    for decision, (now, vehicle, _, row) in zip(monitor.decisions, samples):
        expected = (vehicle, row["cycle"], row["sample_start_s"], row["scheduled_s"], now,
                    row["model"], row["period_ms"])
        actual = tuple(decision[k] for k in ("vehicle", "cycle", "sample_s", "scheduled_s", "decision_s",
                                           "actual_model", "actual_period_ms"))
        matched += actual == expected
    unique = len({(d["vehicle"], d["cycle"]) for d in monitor.decisions})
    if matched != len(samples) or len(monitor.decisions) != len(samples) or unique != len(samples):
        raise ValueError("joint replay request alignment failed")
    monitor.alignment = {"status": "PASS" if samples else "NO_SAMPLES", "expected_samples": len(samples),
                         "matched_samples": matched, "unique_vehicle_cycle_pairs": unique, "mismatches": 0}
    return monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--state-budget-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    profile, config = load_joint_profile(args.summary), load_config(args.state_budget_config)
    sources = [args.report, args.report.parent / "requests.jsonl", args.report.parent / "flight_trace.json"]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    report, traces = read_json(sources[0]), read_json(sources[2])
    requests = [json.loads(line) for line in sources[1].read_text(encoding="utf-8").splitlines()]
    local_gates, frame_source = restore_local_gates(report, traces)
    from .probe_airsim_three import GATE
    monitor = replay(profile, config, report, requests, traces, local_gates, report.get("gate_geometry", GATE))
    path = args.output_dir
    if path is None:
        path = new_output("joint_selection_shadow")
    else:
        path.mkdir(parents=True, exist_ok=False)
    summary = monitor.save(path, execution_mode="causal_replay", source_status=report["status"])
    if hashes != {str(p.resolve()): sha256(p) for p in sources}:
        raise RuntimeError("source flight artifacts changed during replay")
    summary.update(source_sha256=hashes, original_source_acceptance_passed=report["acceptance_passed"],
        original_report_rewritten=False, gate_frame_source=frame_source,
        observer_source_sha256=sha256(Path(__file__)),
        selector_source_sha256=sha256(Path(__file__).with_name("state_joint_policy.py")),
        config_sha256=sha256(args.state_budget_config), gpu_inference_started=False, airsim_started=False)
    atomic_json(path / "joint_shadow_report.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256", "budget_config", "limits")}, indent=2))
    print(f"Report: {path / 'joint_shadow_report.json'}")


if __name__ == "__main__":
    main()
