"""Resource-feedback model/T selection using measured closed-loop candidate costs.

The route cost is a frozen prior, not a state-conditioned safety/performance model.
This policy can keep the smallest model; it never requires a complexity switch.
"""

from pathlib import Path

import numpy as np

from .decision_dataset import sha256
from .dynamic_probe_common import DEFAULT_MODELS, provenance, read_json
from .joint_controller_sweep import DEFAULT_OUTPUT


DEFAULT_SUMMARY = DEFAULT_OUTPUT / "summary.json"


def load_profile(summary_path):
    summary_path = Path(summary_path).resolve()
    suite = read_json(summary_path)
    if suite["status"] != "COMPLETE" or len(suite["trials"]) != suite["planned_trials"]:
        raise ValueError("complete the joint model/period pilot first")
    evidence = provenance(DEFAULT_MODELS, validation=True)
    if suite["provenance"] != evidence:
        raise ValueError("model/data provenance differs from the flight pilot")
    candidates, costs, jitters = [], {}, []
    for trial in suite["trials"]:
        folder = Path(trial["report"]).parent
        for name, expected in trial["artifact_sha256"].items():
            if sha256(folder/name) != expected:
                raise ValueError(f"flight artifact changed: {folder/name}")
        if not trial["eligible"]:
            continue
        if trial["period_ms"] != trial["deadline_ms"]:
            raise ValueError("candidate deadline must equal period")
        report = read_json(folder/"report.json")
        name = trial["model"]
        costs[name] = max(costs.get(name, 0.), report["timing"]["service_ms"]["p95"])
        jitters.append(report["timing"]["release_jitter_ms"]["p95"])
        candidates.append({k: trial[k] for k in ("case", "model", "period_ms", "tracking_mse_m2")})
    if not candidates:
        raise ValueError("no observed feasible candidate in the flight pilot")
    return {"summary": str(summary_path), "summary_sha256": sha256(summary_path),
            "provenance": evidence, "runtime": suite["runtime"], "device": suite["config"]["device"],
            "candidates": candidates, "initial_service_p95_ms": costs,
            "initial_release_jitter_p95_ms": max(jitters),
            "quality_scope": "one validation route; frozen time-integrated tracking MSE, not online flight quality"}


class ResourceSelector:
    def __init__(self, profile, *, utilization_limit=.75, dwell_seconds=2.):
        self.candidates = profile["candidates"]
        self.costs = dict(profile["initial_service_p95_ms"])
        self.prior_costs = dict(self.costs)
        self.cost_updated = dict.fromkeys(self.costs)
        self.jitter = profile["initial_release_jitter_p95_ms"]
        self.utilization_limit = utilization_limit
        self.dwell_seconds = dwell_seconds
        self.current, self.last_switch = None, -float("inf")
        self.static = min(self.candidates, key=lambda c: c["tracking_mse_m2"])

    def select(self, active_clients, now, rows, mode="adaptive"):
        if mode not in ("fixed", "period_only", "adaptive"):
            raise ValueError("unknown resource policy")
        # Only completed, past measurements inform service estimates. No labels,
        # future requests, or future workload stages are available to this method.
        # Live callers provide a bounded snapshot. Small synthetic row lists are
        # retained for unit tests, not used to scan the service's full archive.
        records = rows["recent"] if isinstance(rows, dict) else rows
        recent = [r for r in records if r.get("status") == "ready" and 0 <= now-r["ready_s"] <= 2.]
        for name in self.costs:
            samples = [r["service_ms"] for r in recent if r["model"] == name]
            if len(samples) >= 8:
                self.costs[name] = float(np.percentile(samples, 95))
                self.cost_updated[name] = max(r["ready_s"] for r in recent if r["model"] == name)
        jitter = [(r["sample_start_s"]-r["scheduled_s"])*1000 for r in recent]
        if len(jitter) >= 8:
            self.jitter = max(0., float(np.percentile(jitter, 95)))
        ages = {name: None if stamp is None else max(0., now-stamp)
                for name, stamp in self.cost_updated.items()}
        costs = {name: cost if ages[name] is not None and ages[name] <= 2.
                 else max(cost, self.prior_costs[name]) for name, cost in self.costs.items()}
        pending_ms = (sum(costs[name]*count for name, count in rows["pending_model_counts"].items())
                      if isinstance(rows, dict) else
                      sum(costs[r["model"]] for r in rows if "status" not in r))
        candidates = self.candidates
        if mode == "fixed":
            candidates = [self.static]  # Do not search alternatives in a fixed baseline.
        elif mode == "period_only":
            candidates = [c for c in candidates if c["model"] == self.static["model"]]
        options = []
        for candidate in candidates:
            service_ms = costs[candidate["model"]]
            load = active_clients*service_ms/candidate["period_ms"]
            response_ms = self.jitter+pending_ms+service_ms
            options.append({**candidate, "estimated_load": load, "estimated_response_ms": response_ms,
                            "estimated_feasible": load <= self.utilization_limit
                                and response_ms <= candidate["period_ms"]})
        feasible = [c for c in options if c["estimated_feasible"]]
        if mode == "fixed":
            chosen = next(c for c in options if c["case"] == self.static["case"])
            reason = "fixed_best_observed_route_config"
        elif feasible:
            chosen = min(feasible, key=lambda c: (c["tracking_mse_m2"], c["estimated_load"]))
            reason = "best_prior_quality_within_observed_resource_budget"
            current = next((c for c in feasible if c["case"] == self.current), None)
            if current and chosen["case"] != current["case"]:
                if now-self.last_switch < self.dwell_seconds:
                    chosen, reason = current, "minimum_dwell"
                elif chosen["tracking_mse_m2"] >= current["tracking_mse_m2"]*.99:
                    chosen, reason = current, "quality_gain_below_one_percent"
        else:
            # Replay continues to measure overload; this is NOT a flight fallback.
            chosen = min(options, key=lambda c: (c["estimated_load"], c["estimated_response_ms"]))
            reason = "no_estimated_feasible_candidate_replay_best_effort"
        changed = self.current is not None and self.current != chosen["case"]
        if self.current != chosen["case"]:
            self.last_switch = now
        self.current = chosen["case"]
        return {**chosen, "active_clients": active_clients, "changed": changed, "reason": reason,
                "policy_mode": mode, "service_p95_ms": costs,
                "cost_estimate_age_seconds": ages,
                "cost_estimate_source": {name: "flight_prior" if ages[name] is None else
                    ("recent_observation" if ages[name] <= 2. else "stale_conservative") for name in costs},
                "release_jitter_p95_ms": self.jitter,
                "pending_service_estimate_ms": pending_ms,
                "state_dependent_T_max": None, "is_safety_certificate": False}
