"""Replay the retained advanced-clock failure without new physical observations."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .decision_dataset import atomic_json, sha256
from .framework_runtime import bounded_plan


def audit(directory, output):
    decisions = json.loads((directory/"decisions.json").read_text())
    decision = decisions[-1]
    if "forecast_pending" not in decision or "frontend_reserved_until_s" in decision:
        raise ValueError("select an old failed snapshot that advanced its clock, not a new v9 run")
    report = json.loads((directory/"report.json").read_text())
    costs = report["costs"]
    profile = json.loads((directory/"worker_profile.json").read_text())
    actual,advanced = decision["started_s"],decision["planning_time_s"]
    if advanced < actual:
        raise ValueError("no advanced snapshot clock in this trial")
    pending = deepcopy(decision["forecast_pending"])
    changes = []
    for row in pending:
        for key,bound in (("sample_io_remaining_s","sample_io_s"),("io_remaining_s","io_s"),
                ("command_remaining_s","command_s"),("ingress_remaining_s","ingress_s")):
            if key in row and (key != "ingress_remaining_s" or row[key]>0):
                old = row[key]
                row[key] = min(costs[bound],old+advanced-actual)
                changes.append({"sample_id":row["sample_id"],"field":key,"old_s":old,"corrected_s":row[key]})
    future = {r["vehicle"]:deepcopy(r) for r in decision["forecast_future"]}
    originals = deepcopy([future,pending])
    proposal,info = bounded_plan(actual,future,pending,profile,{**costs,"frontend_reserved_until_s":advanced},
        inflight_limit=report["actual_vehicles"])
    result = {"trial":str(directory.resolve()),"source_sha256":{n:sha256(directory/n)
        for n in ("decisions.json","report.json","worker_profile.json")},
        "old_snapshot_s":advanced,"actual_begin_s":actual,"frontend_reserved_until_s":advanced,
        "remaining_cost_corrections":changes,"new_prediction_feasible":proposal is not None,
        "new_prediction":info,"same_deadlines_and_costs":True,"inputs_unchanged":[future,pending]==originals,
        "new_future_observations_used":False,"offline_prediction_only":True,"new_flight_performed":False,
        "scope":"Rewind the stale snapshot to the recorded planner begin; reserve CPU instead of inventing negative in-flight residuals."}
    atomic_json(output,result)
    print(json.dumps({k:result[k] for k in ("new_prediction_feasible","same_deadlines_and_costs",
        "inputs_unchanged","remaining_cost_corrections")},indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    audit(args.trial,args.output)
