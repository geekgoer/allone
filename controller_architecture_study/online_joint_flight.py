"""Single-policy online model/T takeover on the manual-Play three-drone harness.

Request k keeps its committed model, D_k=T_k and next release t_k+T_k.
The current state selects request k+1; commit only after timely dispatch of k.
No MPC rescue, deadline extension or synthetic workload is used.
"""

import argparse
from collections import deque
from pathlib import Path
import threading
import time

from .decision_dataset import sha256
from .dynamic_probe_common import read_json
from .online_joint_policy import DEFAULT_SUMMARY
from .state_budget_shadow import DEFAULT_CONFIG, gate_budget, load_config
from .state_joint_policy import StateResourceSelector, load_joint_profile


def choice(candidate):
    return {"model": candidate["model"], "period_ms": candidate["period_ms"],
            "deadline_ms": candidate["period_ms"]}


class OnlineJointController:
    def __init__(self, profile, config, service, vehicles, *, mode="adaptive"):
        self.selectors = {v: StateResourceSelector(profile, config) for v in vehicles}
        self.initial = choice(next(iter(self.selectors.values())).resource.static)
        self.next_choices, self.committed_cycles = {}, {}
        self.config, self.service, self.mode = config, service, mode
        self.history = deque(maxlen=config["response_history_size"])
        self.history_lock = threading.Lock()

    def releases(self, start, duration, phase, stop, service, vehicle):
        elapsed, cycle, current = phase, 0, dict(self.initial)
        while elapsed < duration-1e-9:
            scheduled = start+elapsed
            if stop.wait(max(0., scheduled-time.perf_counter())):
                return
            if time.perf_counter() >= scheduled+current["deadline_ms"]/1000:
                service.record_skip(vehicle, cycle, scheduled, **current)
                raise RuntimeError("committed release deadline expired; no catch-up control")
            yield cycle, scheduled, dict(current)
            if stop.is_set():
                return
            if self.committed_cycles.get(vehicle) != cycle:
                raise RuntimeError("next choice was not committed by a timely dispatch")
            elapsed += current["period_ms"]/1000
            current = dict(self.next_choices[vehicle])
            cycle += 1

    def select_next(self, row, raw, gates, geometry):
        row["joint_online_start_s"] = time.perf_counter()
        row["joint_online_sample"] = {"position": raw.position.tolist(), "velocity": raw.velocity.tolist()}
        try:
            state = gate_budget(raw.position, raw.velocity, gates, geometry, self.config)
            snapshot = self.service.feedback_snapshot(time.perf_counter())
            with self.history_lock:
                history = list(self.history)
            now = time.perf_counter()
            if snapshot["inference_error"]:
                raise RuntimeError("shared inference worker reported an error")
            selector = self.selectors[row["vehicle"]]
            observation = selector.select(len(self.selectors), now, snapshot, state,
                mode=self.mode, producer_elapsed_ms=1000*(now-row["sample_start_s"]),
                release_elapsed_ms=1000*(now-row["scheduled_s"]), dispatch_history=history,
                committed_period_ms=row["period_ms"], commit=False)
            checked_at = row["joint_online_ready_s"] = time.perf_counter()
            row["joint_online_decision"], row["joint_online_state"] = observation, state
            if observation["recommendation"] is None:
                raise RuntimeError(f"no eligible next model/T: {observation['status']}")
            row["joint_online_initial_case"] = observation["recommendation"]["case"]
            # Costs only grow after this observation. Re-screen the frozen
            # options, including measured re-screen time; never extend D_k.
            for attempt in range(len(observation["candidate_options"])+1):
                if checked_at >= row["scheduled_s"]+row["deadline_ms"]/1000:
                    raise RuntimeError("online selection exceeded the immutable current deadline")
                decision = selector.charge_selection(observation, checked_at,
                    selection_tail_ms=1000*(checked_at-now),
                    release_elapsed_ms=1000*(checked_at-row["scheduled_s"]),
                    committed_period_ms=row["period_ms"])
                ready = row["joint_online_ready_s"] = time.perf_counter()
                row["joint_online_decision"] = decision
                selected = decision["recommendation"]
                if selected is None:
                    raise RuntimeError(f"no eligible next model/T: {decision['status']}")
                response = selected["estimated_release_response_ms"]+1000*(ready-checked_at)
                age = row["period_ms"]+response
                row["joint_online_admission"] = {"selection_tail_ms": 1000*(ready-now),
                    "charged_selection_tail_ms": decision["charged_selection_tail_ms"],
                    "rescreens": attempt+1,
                    "estimated_release_response_with_selection_ms": response,
                    "estimated_age_with_selection_ms": age}
                if ready >= row["scheduled_s"]+row["deadline_ms"]/1000:
                    raise RuntimeError("online selection exceeded the immutable current deadline")
                if response <= selected["period_ms"] and (not state["applicable"] or age <= state["validity_budget_ms"]):
                    break
                checked_at = ready
            else:
                raise RuntimeError("selection overhead exhausted the predicted next-response budget")
            row.update(next_model=selected["model"], next_period_ms=selected["period_ms"],
                       next_deadline_ms=selected["period_ms"])
        except Exception as exc:
            if "joint_online_ready_s" not in row:
                row["joint_online_ready_s"] = time.perf_counter()
            row.update(control_outcome="policy_abort", error=f"{type(exc).__name__}: {exc}")
            self.service.record_abort(row)
            raise

    def commit_dispatch(self, row):
        if row["dnn_dispatch_s"] > row["scheduled_s"]+row["deadline_ms"]/1000:
            raise RuntimeError("async command dispatch exceeded the immutable current deadline")
        record = {k: row[k] for k in ("enqueue_s", "sample_start_s", "ready_s", "dnn_dispatch_s")}
        with self.history_lock:
            self.history.append(record)
        self.next_choices[row["vehicle"]] = {"model": row["next_model"],
            "period_ms": row["next_period_ms"], "deadline_ms": row["next_deadline_ms"]}
        self.committed_cycles[row["vehicle"]] = row["cycle"]
        self.selectors[row["vehicle"]].commit(row["joint_online_decision"]["recommendation"], row["dnn_dispatch_s"])
        row["next_choice_committed"] = True


def audit_schedule(rows, vehicles, start, duration, phase, initial):
    """Audit the realized dynamic schedule, including its final release horizon."""
    failures = []
    for i, vehicle in enumerate(vehicles):
        elapsed = i*initial["period_ms"]/3000 if phase == "staggered" else 0.
        expected = dict(initial)
        records = sorted((r for r in rows if r["vehicle"] == vehicle), key=lambda r: r["cycle"])
        for cycle, row in enumerate(records):
            if (start is None or elapsed >= duration-1e-9 or row["cycle"] != cycle
                    or abs(row["scheduled_s"]-(start+elapsed)) > 1e-6
                    or any(row.get(k) != expected[k] for k in expected)
                    or row.get("command_duration_ms") != row["period_ms"]
                    or row.get("next_deadline_ms") != row.get("next_period_ms")
                    or not row.get("next_choice_committed")):
                failures.append(f"{vehicle}: invalid release/commit at cycle {cycle}")
                break
            elapsed += row["period_ms"]/1000
            expected = {"model": row["next_model"], "period_ms": row["next_period_ms"],
                        "deadline_ms": row["next_deadline_ms"]}
        if elapsed < duration-1e-9:
            failures.append(f"{vehicle}: release horizon incomplete")
    return {"passed": not failures, "realized_releases": len(rows), "errors": failures,
            "scope": "contiguous committed model/T/D, command duration and full release horizon"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--state-budget-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", choices=("fixed", "period_only", "adaptive"), default="adaptive")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    profile = args.joint_profile = load_joint_profile(args.summary)
    load_config(args.state_budget_config)
    if args.device != profile["device"]:
        parser.error("use the same device type as the flown candidate profile")
    pilot = read_json(args.summary)
    from .probe_airsim_three import load_layout, run
    _, route_hash = load_layout(pilot["config"]["environment_id"], pilot["config"]["route_index"])
    if route_hash != pilot["saved_route_sha256"]:
        raise ValueError("pilot route changed")
    initial = choice(min(profile["candidates"], key=lambda c: c["tracking_mse_m2"]))
    if args.preflight:
        import json
        print(json.dumps({"status": "PASS", "policy": args.policy, "initial": initial,
            "eligible_pairs": len(profile["candidates"]), "profile_sha256": profile["summary_sha256"],
            "budget_config_sha256": sha256(args.state_budget_config),
            "scope": "next-request online selection; current D=T and next release immutable",
            "gpu_inference_started": False, "airsim_started": False}, indent=2))
        return
    args.joint_online = True
    args.mode, args.phase, args.preprocess_policy = "dnn", "staggered", "vectorized"
    args.environment_id, args.route_index = pilot["config"]["environment_id"], pilot["config"]["route_index"]
    args.model, args.period_ms, args.deadline_ms = initial["model"], initial["period_ms"], initial["deadline_ms"]
    run(args)


if __name__ == "__main__":
    main()
