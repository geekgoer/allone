"""One fixed-20ms counterexample check for the frozen seven-UAV residency trial."""

import sys

from .decision_dataset import ROOT, atomic_json
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_world import FrameworkWorld
from .inflight_sensing import InflightSensing
from .probe_frontend_order import normal_metrics
from .probe_inflight_events import event_outcomes
from .probe_seven_continuous_timing import flight, matched_initials


def run(placement):
    root = ROOT/"controller_architecture_study/results/feedback_residency_20261027_v2/fixed20"
    root.mkdir(parents=True, exist_ok=False)
    source = root.parent/"full_sequence"
    prior = read_json(source/"plan.json")
    pool = ControllerPool()
    if pool.provenance() != prior["provenance"] or placement.layout != prior["cpu_layout"]:
        raise ValueError("model/resource domain changed")
    scenario = prior["prior_plan"]
    world = FrameworkWorld(scenario["layout"], scenario["phase_s"], 7, 2.,
        phase_span=2.1, task_duration=scenario["duration_s"])
    pulse = dict(vehicle="Drone1", time_s=1.207, delta_y_mps=1.95)
    atomic_json(root/"plan.json", dict(period_ms=20., source=str(source),
        source_plan=prior, pulse=pulse, resident_worker=True,
        purpose="test fixed faster control; do not claim an offline-optimal FALCON comparison"))
    rows = []
    try:
        for name, event in (("normal_static", None), ("r1_static", pulse), ("r2_static", pulse)):
            print("FIXED20", name, flush=True)
            d = root/name
            sensing = InflightSensing(world.vehicles, world.calibration, placement.layout["io_cpus"][1], event)
            report = flight(world, pool, 20., dict(id=name, critical_index=-1), "static", d,
                scenario["duration_s"], placement.layout, sensing=sensing, frontend_order="category",
                trace_frontend=True, resident_worker=True)
            outcome = event_outcomes(report, d, event)
            metrics = normal_metrics(report, read_json(d/"requests.json"), period=.020) if event is None else None
            rows.append(dict(name=name, report=report, event_outcomes=outcome, normal_metrics=metrics))
            atomic_json(root/"progress.json", dict(complete=False, rows=rows))
            print("success", report["success"], "collisions", report["collisions"],
                "late", len(report["nominal_late_jobs"]), "fresh_ms", outcome.get("first_fresh_control_after_impulse_ms"), flush=True)
            if report["error"] or report["cleanup_errors"]:
                raise RuntimeError("technical failure retained; no retry")
        initial = read_json(source/"normal_static/initial_state.json")
        atomic_json(root/"summary.json", dict(complete=True, rows=rows,
            initial_states_match=all(matched_initials(initial, read_json(root/r["name"]/"initial_state.json")) for r in rows),
            FPGA_verified=False, not_full_FALCON=True))
    finally:
        atomic_json(root/"cleanup_report.json", world.close())


def main():
    sys.setswitchinterval(.0005)
    placement = CpuPlacement()
    try:
        with placement:
            run(placement)
    finally:
        root = ROOT/"controller_architecture_study/results/feedback_residency_20261027_v2/fixed20"
        if root.exists():
            atomic_json(root/"cpu_placement.json", placement.report)


if __name__ == "__main__":
    main()
