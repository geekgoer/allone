"""Conditional state/resource model-T recommendations, not a flight fallback.

Legacy resource-only replay remains unchanged. A recommendation may be absent;
the caller must never treat that as permission to apply an infeasible output.
"""

from pathlib import Path
import json

import numpy as np

from .dynamic_probe_common import read_json
from .online_joint_policy import ResourceSelector, load_profile


def load_joint_profile(summary_path):
    profile = load_profile(summary_path)  # Also verifies frozen model/flight hashes.
    producer, dispatch = [], []
    for trial in read_json(summary_path)["trials"]:
        if not trial["eligible"]:
            continue
        path = Path(trial["report"]).parent / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        before, after = [], []
        for row in rows:
            if "dnn_dispatch_s" not in row:
                continue
            before.append(1000*(row["enqueue_s"]-row["sample_start_s"]))
            after.append(1000*(row["dnn_dispatch_s"]-row["ready_s"]))
        if not before or not np.isfinite(before+after).all() or min(before+after) < 0:
            raise ValueError("pilot lacks valid end-to-end dispatch timing")
        producer.append(float(np.percentile(before, 95)))
        dispatch.append(float(np.percentile(after, 95)))
    return {**profile, "initial_producer_p95_ms": max(producer),
            "initial_dispatch_p95_ms": max(dispatch)}


class StateResourceSelector:
    def __init__(self, profile, config, *, dwell_seconds=2.):
        self.resource = ResourceSelector(profile, dwell_seconds=dwell_seconds)
        self.candidates = profile["candidates"]
        self.producer_prior = profile["initial_producer_p95_ms"]
        self.dispatch_prior = profile["initial_dispatch_p95_ms"]
        self.config, self.dwell_seconds = config, dwell_seconds
        self.current, self.last_switch = None, -float("inf")

    def select(self, active_clients, now, snapshot, state, *, mode="adaptive",
               producer_elapsed_ms=0., release_elapsed_ms=0., dispatch_history=(),
               committed_period_ms=None, commit=True):
        if mode not in ("fixed", "period_only", "adaptive"):
            raise ValueError("unknown joint policy")
        if active_clients <= 0 or not np.isfinite([now, producer_elapsed_ms, release_elapsed_ms]).all():
            raise ValueError("invalid decision time or active client count")
        if min(producer_elapsed_ms, release_elapsed_ms) < 0:
            raise ValueError("decision cannot precede sampling/release")
        if committed_period_ms is not None and (not np.isfinite(committed_period_ms) or committed_period_ms <= 0):
            raise ValueError("committed period must be positive and finite")
        # Use legacy bounded resource estimation, but NOT its best-effort choice.
        resource = self.resource.select(active_clients, now, snapshot, mode="fixed")
        past = sorted((r for r in dispatch_history if r["dnn_dispatch_s"] < now),
                      key=lambda r: r["dnn_dispatch_s"])
        recent = [r for r in past if now-r["dnn_dispatch_s"] <= 2.]
        producer_ms, dispatch_ms = max(self.producer_prior, producer_elapsed_ms), self.dispatch_prior
        if len(recent) >= 8:
            producer_ms = max(producer_ms, float(np.percentile(
                [1000*(r["enqueue_s"]-r["sample_start_s"]) for r in recent], 95)))
            dispatch_ms = max(dispatch_ms, float(np.percentile(
                [1000*(r["dnn_dispatch_s"]-r["ready_s"]) for r in recent], 95)))
        history = past[-self.config["response_history_size"]:]
        reserve = max(self.config["response_floor_ms"], self.config["response_margin_ms"] +
            max((1000*(r["dnn_dispatch_s"]-r["sample_start_s"]) for r in history), default=0.))
        applicable = bool(state["applicable"])
        invalid = applicable and bool(state["observed_rate_bound_exceeded"])
        budget = state["validity_budget_ms"] if applicable else None
        if applicable and (budget is None or not np.isfinite(budget)):
            raise ValueError("applicable state needs a finite validity budget")
        candidates = self.candidates
        if mode == "fixed":
            candidates = [self.resource.static]
        elif mode == "period_only":
            candidates = [c for c in candidates if c["model"] == self.resource.static["model"]]
        options = []
        for candidate in candidates:
            service_ms = resource["service_p95_ms"][candidate["model"]]
            load = active_clients*service_ms/candidate["period_ms"]
            response_ms = max(reserve, producer_ms + resource["pending_service_estimate_ms"] +
                service_ms + dispatch_ms + self.config["response_margin_ms"])
            release_response_ms = resource["release_jitter_p95_ms"] + response_ms
            # Candidate T is a nominal release interval. Reserve next-release
            # jitter too, rather than silently identifying it with actual T_k.
            # Online selection is for request k+1. Its release is already
            # committed by T_k; selecting T_(k+1) cannot move that release.
            interval_ms = candidate["period_ms"] if committed_period_ms is None else committed_period_ms
            age_ms = interval_ms + release_response_ms
            resource_ok = load <= self.resource.utilization_limit and release_response_ms <= candidate["period_ms"]
            committed_in_time = release_elapsed_ms < interval_ms
            state_ok = None if not applicable or invalid else age_ms <= budget
            options.append({**candidate, "estimated_load": load,
                "estimated_sample_response_ms": response_ms,
                "estimated_release_response_ms": release_response_ms,
                "estimated_age_at_next_dispatch_ms": age_ms,
                "estimated_nominal_T_max_ms": None if budget is None else budget-release_response_ms,
                "resource_feasible": resource_ok, "next_release_not_elapsed": committed_in_time,
                "state_feasible": state_ok,
                "estimated_feasible": resource_ok and committed_in_time and not invalid and state_ok is not False})
        decision = {"policy_mode": mode, "active_clients": active_clients,
            "state_applicable": applicable, "assumption_invalid": invalid,
            "validity_budget_ms": budget, "candidate_options": options,
            "service_p95_ms": resource["service_p95_ms"],
            "cost_estimate_source": resource["cost_estimate_source"],
            "release_jitter_p95_ms": resource["release_jitter_p95_ms"],
            "pending_service_estimate_ms": resource["pending_service_estimate_ms"],
            "producer_p95_reserve_ms": producer_ms, "dispatch_p95_reserve_ms": dispatch_ms,
            "completed_response_reserve_ms": reserve, "is_safety_certificate": False}
        return self._recommend(decision, now, commit=commit)

    def charge_selection(self, decision, now, *, selection_tail_ms, release_elapsed_ms,
                         committed_period_ms):
        """Recheck every next-request option against measured selection cost.

        Always pass the original observation, so retry costs are not added twice.
        This preview does not change committed choice/dwell history.
        """
        if (not np.isfinite([now, selection_tail_ms, release_elapsed_ms, committed_period_ms]).all()
                or min(selection_tail_ms, release_elapsed_ms) < 0 or committed_period_ms <= 0):
            raise ValueError("invalid selection admission timing")
        budget, options = decision["validity_budget_ms"], []
        for candidate in decision["candidate_options"]:
            response = candidate["estimated_release_response_ms"] + selection_tail_ms
            age = committed_period_ms + response
            resource_ok = (candidate["estimated_load"] <= self.resource.utilization_limit
                           and response <= candidate["period_ms"])
            in_time = release_elapsed_ms < committed_period_ms
            state_ok = None if not decision["state_applicable"] or decision["assumption_invalid"] else age <= budget
            options.append({**candidate,
                "estimated_sample_response_ms": candidate["estimated_sample_response_ms"]+selection_tail_ms,
                "estimated_release_response_ms": response, "estimated_age_at_next_dispatch_ms": age,
                "estimated_nominal_T_max_ms": None if budget is None else budget-response,
                "resource_feasible": resource_ok, "next_release_not_elapsed": in_time,
                "state_feasible": state_ok,
                "estimated_feasible": resource_ok and in_time and not decision["assumption_invalid"] and state_ok is not False})
        return self._recommend({**decision, "candidate_options": options,
                                "charged_selection_tail_ms": selection_tail_ms}, now, commit=False)

    def _recommend(self, decision, now, *, commit):
        options = decision["candidate_options"]
        feasible = [c for c in options if c["estimated_feasible"]]
        chosen, reason = None, None
        if decision["assumption_invalid"]:
            status = "ASSUMPTION_INVALID"
        elif not feasible:
            status = "BUDGET_EXHAUSTED" if any(c["resource_feasible"] and c["next_release_not_elapsed"] for c in options) else "RESOURCE_EXHAUSTED"
        else:
            status = "AVAILABLE" if decision["state_applicable"] else "STATE_NOT_APPLICABLE"
            chosen = min(feasible, key=lambda c: (c["tracking_mse_m2"], c["estimated_load"]))
            reason = "best_frozen_route_cost_among_joint_feasible_candidates"
            current = next((c for c in feasible if c["case"] == self.current), None)
            if current and current["case"] != chosen["case"]:
                if now-self.last_switch < self.dwell_seconds:
                    chosen, reason = current, "minimum_dwell"
                elif chosen["tracking_mse_m2"] >= current["tracking_mse_m2"]*.99:
                    chosen, reason = current, "quality_gain_below_one_percent"
        changed = chosen is not None and self.current is not None and self.current != chosen["case"]
        if commit and chosen is not None:
            self.commit(chosen, now)
        return {**decision, "status": status, "recommendation": chosen, "changed": changed, "reason": reason}

    def commit(self, chosen, now):
        if self.current != chosen["case"]:
            self.current, self.last_switch = chosen["case"], now
