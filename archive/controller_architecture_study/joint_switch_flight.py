"""Scripted model/T transitions in the existing three-drone closed-loop harness.

This checks the switching mechanism, NOT online resource/state adaptation.
The plan contains previously flown candidate pairs; no model or data is trained.
"""

import argparse
from pathlib import Path
import time

import numpy as np

from .decision_dataset import ROOT, sha256
from .dynamic_probe_common import read_json
from .online_joint_policy import DEFAULT_SUMMARY, load_profile


DEFAULT_PLAN = ROOT / "controller_architecture_study/configs/joint_switch_mechanism_v1.json"


class SwitchPlan:
    def __init__(self, steps, candidates):
        if not steps:
            raise ValueError("switch plan is empty")
        allowed = {(c["model"], c["period_ms"]) for c in candidates}
        previous = -1.
        self.steps = []
        for entry in steps:
            at, period = float(entry["at_seconds"]), float(entry["period_ms"])
            if not np.isfinite([at, period]).all() or at < 0 or at <= previous or period <= 0:
                raise ValueError("switch times must be finite, nonnegative and strictly increasing")
            if (entry["model"], period) not in allowed:
                raise ValueError("switch pair was not an eligible flown candidate")
            self.steps.append({"at_seconds": at, "model": entry["model"],
                               "period_ms": period, "deadline_ms": period})
            previous = at
        if self.steps[0]["at_seconds"] != 0.:
            raise ValueError("the first switch configuration must start at zero")

    def at(self, elapsed):
        return dict(next(step for step in reversed(self.steps) if step["at_seconds"] <= elapsed+1e-8))

    def slots(self, start, duration, phase=0.):
        """Keep the next committed release; never reset/rephase at a switch."""
        # Accumulate relative time, so host uptime cannot change the endpoint
        # count or the configuration chosen at a switch boundary.
        elapsed, cycle = phase, 0
        while elapsed < duration-1e-9:
            choice = self.at(elapsed)
            yield cycle, start+elapsed, choice
            elapsed += choice["period_ms"]/1000
            cycle += 1

    def releases(self, start, duration, phase, stop, service, vehicle):
        for cycle, scheduled, choice in self.slots(start, duration, phase):
            if stop.wait(max(0., scheduled-time.perf_counter())):
                break
            if time.perf_counter() >= scheduled+choice["deadline_ms"]/1000:
                service.record_skip(vehicle, cycle, scheduled, **choice)
                continue
            yield cycle, scheduled, choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--keep-gates", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--state-budget-config", type=Path,
                        help="Post-flight causal T_max shadow replay; no automatic model/T choices")
    parser.add_argument("--joint-selection-shadow", action="store_true",
                        help="Capture states during flight and compare fixed/T-only/joint choices AFTER flight; no control change")
    args = parser.parse_args()
    profile = load_profile(args.summary)
    pilot = read_json(args.summary)
    if args.device != profile["device"]:
        parser.error("use the same device type as the flown candidate profile")
    plan = SwitchPlan(read_json(args.plan)["steps"], profile["candidates"])
    if args.joint_selection_shadow:
        from .state_budget_shadow import DEFAULT_CONFIG
        from .state_joint_policy import load_joint_profile
        if args.state_budget_config is None:
            args.state_budget_config = DEFAULT_CONFIG
        args.joint_profile = load_joint_profile(args.summary)
    if args.state_budget_config is not None:
        from .state_budget_shadow import load_config
        load_config(args.state_budget_config)
    from .probe_airsim_three import load_layout, run
    layout, route_hash = load_layout(pilot["config"]["environment_id"], pilot["config"]["route_index"])
    if route_hash != pilot["saved_route_sha256"]:
        raise ValueError("pilot route changed")
    duration = max(20., float(layout["duration"])+2.)
    if plan.steps[-1]["at_seconds"] >= min(duration, float(layout["duration"])):
        raise ValueError("switches must occur during the active route, not only the hover tail")
    if args.preflight:
        import json
        print(json.dumps({"status": "PASS", "steps": plan.steps,
                          "trajectory_clock": "one continuous host clock",
                          "scope": "scripted mechanism test, not online adaptation",
                          "state_budget_mode": "causal_shadow_replay" if args.state_budget_config else None,
                          "state_budget_config_sha256": sha256(args.state_budget_config) if args.state_budget_config else None,
                          "joint_selection_shadow": args.joint_selection_shadow,
                          "joint_selection_execution": "post_flight_causal_replay" if args.joint_selection_shadow else None,
                          "gpu_inference_started": False, "airsim_started": False}, indent=2))
        return
    args.switch_plan = plan
    args.switch_plan_sha256 = sha256(args.plan)
    args.switch_profile_sha256 = profile["summary_sha256"]
    args.switch_runtime = profile["runtime"]
    args.mode, args.phase, args.preprocess_policy = "dnn", "staggered", "vectorized"
    args.environment_id, args.route_index = pilot["config"]["environment_id"], pilot["config"]["route_index"]
    args.model = plan.steps[0]["model"]
    args.period_ms = args.deadline_ms = plan.steps[0]["period_ms"]
    run(args)


if __name__ == "__main__":
    main()
