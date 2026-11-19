"""Frozen-case mechanism ablations and a bounded normal-only static selection."""

import argparse
from pathlib import Path
import shutil
import sys

from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import ControllerPool
from .framework_runtime_editor import read_background_throttle
from .framework_runtime_world import FrameworkWorld
from .inflight_sensing import InflightSensing
from .probe_frontend_order import normal_metrics
from .probe_inflight_events import event_outcomes
from .probe_seven_continuous_timing import flight, matched_initials, calibrated_response_reserve


SOURCE = ROOT/"controller_architecture_study/results/feedback_residency_20261027_v2/full_sequence"
LARGE = "plain_mlp_h9_w1024_seed0"
SMALL = "plain_mlp_h9_w512_seed0"


def select_static(rows):
    eligible = [r for r in rows if r["report"]["success"]
        and r["normal_metrics"]["effective_misses"] == 0 and r["report"]["total_J"] is not None]
    return max(eligible, key=lambda r:r["report"]["total_J"]) if eligible else None


def run(root, placement):
    root.mkdir(parents=True, exist_ok=False)
    source = read_json(SOURCE/"plan.json")
    scene = source["prior_plan"]
    pool = ControllerPool()
    editor = read_background_throttle()
    if pool.provenance() != source["provenance"] or placement.layout != source["cpu_layout"]:
        raise ValueError("model or CPU domain changed")
    if editor["background_throttle_enabled"]:
        raise RuntimeError("editor background throttling enabled")
    candidates = [(SMALL,40.),(LARGE,20.),(SMALL,20.),(LARGE,40.),(SMALL,30.),(LARGE,30.)]
    pulse = dict(vehicle="Drone1", time_s=1.207, delta_y_mps=1.95)
    frozen = root/"frozen_source"
    frozen.mkdir()
    names = sorted(set(source["source_sha256"]) | {Path(__file__).name, "probe_frontend_order.py"})
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), frozen/name)
    atomic_json(root/"plan.json", dict(source=str(SOURCE), scene=scene, pulse=pulse,
        source_sha256={n:sha256(frozen/n) for n in names}, provenance=pool.provenance(),
        cpu_layout=placement.layout, editor=editor, normal_candidates=candidates,
        fallback_candidates=[(SMALL,60.),(LARGE,60.)], fallback_rule="only if no zero-miss candidate",
        selection="highest normal-flight J with zero missing/late scheduled updates; freeze before pulses",
        variants=["full","no_B","fixed_priority","new_reserve","no_residency"], repeats=2,
        variant_order="listed then reverse; selected static first/last if available",
        minimum_period_ms=20., legacy_reserve_s=source["response_reserve_s"],
        new_reserve_source="normal large T40 capture-to-command p99 * 1.25 + 0.5ms",
        not_full_FALCON=True, not_physical_safety_certificate=True, FPGA_verified=False))
    world = FrameworkWorld(scene["layout"], scene["phase_s"], 7, 2., phase_span=2.1,
        task_duration=scene["duration_s"])
    rows = []
    reference_initial = read_json(SOURCE/"normal_static/initial_state.json")

    def execute(name, *, model=LARGE, period=40., policy="adaptive_twosided", event=None,
                scheduler=None, physical_budget=True, resident=True, reserve=None):
        if (root/"STOP").exists():
            raise RuntimeError("operator STOP requested between trials")
        print("MECHANISM", name, flush=True)
        directory = root/name
        sensing = InflightSensing(world.vehicles, world.calibration, placement.layout["io_cpus"][1], event)
        report = flight(world, pool, period, dict(id=name,critical_index=-1), policy, directory,
            scene["duration_s"], placement.layout, sensing=sensing, response_reserve_s=reserve,
            frontend_order="category", trace_frontend=True, resident_worker=resident,
            model=model, scheduler=scheduler, physical_budget=physical_budget)
        outcome = event_outcomes(report, directory, event)
        metrics = normal_metrics(report, read_json(directory/"requests.json"), period=period/1000) if event is None and policy=="static" else None
        row = dict(name=name, report=report, event_outcomes=outcome, normal_metrics=metrics,
            initial_states_match=matched_initials(reference_initial, read_json(directory/"initial_state.json")))
        rows.append(row)
        atomic_json(root/"progress.json", dict(complete=False,rows=rows))
        print(" success", report["success"], "collisions", report["collisions"],
            "late", len(report["nominal_late_jobs"]), "fresh_ms", outcome.get("first_fresh_control_after_impulse_ms"),
            "missing+late", metrics and metrics["effective_misses"], "J", report["total_J"], flush=True)
        if report["error"] or report["cleanup_errors"] or not row["initial_states_match"]:
            raise RuntimeError("technical/initial-state mismatch retained; no retry")
        return row

    try:
        normal = []
        for model,period in candidates:
            normal.append(execute(f"normal_{model}_T{period:g}", model=model, period=period, policy="static"))
        selected = select_static(normal)
        if selected is None:
            for model in (SMALL,LARGE):
                normal.append(execute(f"normal_{model}_T60", model=model, period=60., policy="static"))
            selected = select_static(normal)
        chosen = None if selected is None else dict(model=selected["report"]["model"], period=selected["report"]["period_ms"])
        atomic_json(root/"selection.json", dict(selected=chosen, source_trial=selected and selected["name"],
            frozen_before_challenges=True, strict_normal_candidate_found=selected is not None,
            not_full_FALCON=True, normal_trials=[dict(name=r["name"],J=r["report"]["total_J"],
                effective_misses=r["normal_metrics"]["effective_misses"]) for r in normal]))
        confirmation = None if chosen is None else execute("normal_selected_confirmation", policy="static", **chosen)
        old_reserve = source["response_reserve_s"]
        calibration = next(r for r in normal if r["report"]["model"]==LARGE and r["report"]["period_ms"]==40.)
        new_reserve = calibrated_response_reserve(calibration["report"])
        variants = [("full",{}),("no_B",dict(physical_budget=False)),
            ("fixed_priority",dict(scheduler="static")),("new_reserve",dict(reserve=new_reserve)),
            ("no_residency",dict(resident=False))]
        if chosen is not None:
            variants.insert(0,("selected_static",dict(policy="static",**chosen)))
        atomic_json(root/"ablation_plan.json", dict(variants=variants, old_reserve_s=old_reserve,
            new_reserve_s=new_reserve, selection_confirmation=confirmation and dict(
                success=confirmation["report"]["success"], effective_misses=confirmation["normal_metrics"]["effective_misses"])))
        for repeat in (1,2):
            for label, options in variants if repeat==1 else reversed(variants):
                execute(f"r{repeat}_{label}", event=pulse, **{"reserve":old_reserve, **options})
        challenged = [r for r in rows if r["name"].startswith("r")]
        times = [r["event_outcomes"]["injection"]["post_simulation_s"] for r in challenged]
        atomic_json(root/"summary.json", dict(complete=True, rows=rows, selected=chosen,
            initial_states_match=all(r["initial_states_match"] for r in rows),
            pulses_valid_and_matched=all(r["event_outcomes"]["disturbance_valid"] for r in challenged)
                and max(times)-min(times)<=.012, old_reserve_s=old_reserve, new_reserve_s=new_reserve,
            strict_static_confirmation_passed=confirmation is not None and confirmation["report"]["success"]
                and confirmation["normal_metrics"]["effective_misses"]==0,
            not_full_FALCON=True, FPGA_verified=False))
    finally:
        atomic_json(root/"cleanup_report.json", world.close())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.setswitchinterval(.0005)
    placement = CpuPlacement()
    try:
        with placement:
            run(args.output, placement)
    finally:
        if args.output.exists():
            atomic_json(args.output/"cpu_placement.json", placement.report)


if __name__=="__main__":
    main()
