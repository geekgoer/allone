"""Keep every validation-context outcome and distinguish teacher/task limitations."""

import csv
from pathlib import Path
import sys

import numpy as np

from .decision_dataset import atomic_json, sha256
from .state_mode_probe import OUTPUT, read_json
from .state_mode_boundary import BoundaryPool, CONTROLLERS
from .state_mode_generalization import RESULTS
from .analyze_state_mode_boundary import trajectory_metrics, paired_outcomes, exact_tail_sensitivity


def reference_control_diagnostic(plan):
    """Ideal-reference queries, not a changed controller or a causal flight test."""
    from airsim_interface import RawDroneState
    from .convex_teacher import ConvexTeacher
    from .fast_convex_preprocess import FastReferenceWindow
    from .fly_decision_dnn import StoredRoutePlanner
    from .probe_airsim_three import load_layout
    from .state_mode_calibration import moderate_case

    teacher = ConvexTeacher()
    result = {}
    prior, _ = moderate_case()
    contexts = plan["contexts"]+[{"id": "prior_e48_r1", "environment": 48, "route": 1,
                                  "phase_s": prior["reference_phase_s"]}]
    for context in contexts:
        layout, _ = load_layout(context["environment"], context["route"])
        planner = StoredRoutePlanner(layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        rows = []
        for t in np.arange(0., 3.01, .05):
            refs = window.window(context["phase_s"]+t)
            ref = refs[0]
            state = RawDroneState(ref.position, ref.orientation, ref.velocity, np.zeros(3), np.zeros(3))
            action, info = teacher.query(state, refs)
            rows.append({"t": float(t), "reference_acceleration": ref.acceleration.tolist(),
                "QP_first_acceleration": info["first_acceleration"].tolist(),
                "reference_body_rates": ref.body_rates.tolist(), "teacher_body_rates": action[1:].tolist(),
                "body_rate_difference_norm_rps": float(np.linalg.norm(ref.body_rates-action[1:])),
                "prediction_constraint_active": bool(info["acceleration_bound_active"] or info["tilt_bound_active"])})
        result[context["id"]] = {"rows": rows,
            "reference_context": context,
            "max_body_rate_difference_norm_rps": max(r["body_rate_difference_norm_rps"] for r in rows),
            "mean_body_rate_difference_norm_rps": float(np.mean([r["body_rate_difference_norm_rps"] for r in rows])),
            "queries_with_active_prediction_constraints": sum(r["prediction_constraint_active"] for r in rows)}
    return {"contexts": result,
        "scope": "Offline at exact reference p/v/q: compare reference kinematic angular rates with actual teacher labels. "
                 "A discrepancy means the reference is not exactly reproduced at zero tracking error under an ideal rate actuator. "
                 "Not proof that missing feedforward alone causes flight errors, not an implemented controller fix.",
        "controller_modified": False, "test_read": False}


def initial_alignment(plan):
    """Matched initial queries only; never used as an online decision feature."""
    from airsim_interface import RawDroneState
    from .fast_convex_preprocess import FastReferenceWindow
    from .fly_decision_dnn import StoredRoutePlanner
    from .probe_airsim_three import load_layout
    original = read_json(OUTPUT/"setup.json")["plan"]
    pool = BoundaryPool(original)
    std = np.asarray(read_json(Path(original["export_dir"])/"normalization.json")["u_std"])
    rows = []
    for case in plan["cases"]:
        if case["kind"] != "recovery" or case["period_ms"] != 50.:
            continue
        layout, _ = load_layout(case["environment_id"], case["route_index"])
        planner = StoredRoutePlanner(layout)
        refs = FastReferenceWindow(planner.trajectory, planner.config).window(case["reference_phase_s"])
        state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
        feature = pool.teacher.encode(state, refs)
        actions = {mode: pool.predict(mode, feature)[0] for mode in CONTROLLERS}
        rows.append({"case": case["source_case"], "physical_outputs": {m: a.tolist() for m, a in actions.items()},
            "teacher_normalized_mse": {m: float(np.mean(((actions[m]-actions["QP_teacher"])/std)**2))
                                       for m in CONTROLLERS[1:]}})
    return {"rows": rows, "scope": "Six initial states, not dataset-wide error or closed-loop capability; "
            "teacher comparisons are offline only, not an added runtime error estimator", "test_read": False}


def main():
    setup, data = read_json(RESULTS/"setup.json"), read_json(RESULTS/"summary.json")
    plan = setup["plan"]
    if data["completed"] != len(plan["trials"]):
        raise ValueError("complete the frozen plan before final analysis")
    cases = {c["id"]: c for c in plan["cases"]}
    rows, recovery = [], []
    for item in data["trials"]:
        report, tail = item["report"], item["supplementary"]
        trial = report["trial"]
        case = cases[trial["case"]]
        directory = RESULTS/trial["id"]
        trace, commands = read_json(directory/"trace.json"), read_json(directory/"commands.json")
        gaps = [c["command_send_gap_ms"] for c in commands if c["slot"] > 0 and "command_send_gap_ms" in c]
        rows.append({"trial_id": trial["id"], "context": case["context"], "kind": case["kind"],
            "case": case["id"], "profile": case["profile"], "period_ms": case["period_ms"],
            "controller": trial["mode"], "repeat": trial["repeat"],
            "control_success": report["control_success"], "sustained_control": tail["sustained_control_passed"],
            "sustained_joint": tail["sustained_joint_passed"], "timing_comparable": report["timing_comparable"],
            "tail_max_position_error_m": tail["tail_max_position_error_m"],
            "tail_max_velocity_error_mps": tail["tail_max_velocity_error_mps"],
            "minimum_h_m": report["minimum_h_m"], "progress_ratio": report["progress_ratio"],
            "deadline_misses": report["deadline_misses"], "delivery_deviations": report["delivery_deviations"],
            "rpc_returns_after_deadline": report["rpc_returns_after_deadline"],
            "mean_command_send_gap_ms": sum(gaps)/len(gaps) if gaps else None,
            "max_command_send_gap_ms": max(gaps, default=None),
            "error_kind": report["error_kind"], "error": report["error"],
            **trajectory_metrics(trace), **exact_tail_sensitivity(trace)})
        if case["kind"] == "recovery":
            recovery.append(item)
    pairs = paired_outcomes({"trials": recovery})
    totals = {"completed": len(rows), "original_control_success": sum(r["control_success"] for r in rows),
        "sustained_control": sum(r["sustained_control"] for r in rows),
        "sustained_joint": sum(r["sustained_joint"] for r in rows),
        "timing_invalid": sum(not r["timing_comparable"] for r in rows),
        "deadline_misses": sum(r["deadline_misses"] for r in rows),
        "rpc_returns_after_deadline": sum(r["rpc_returns_after_deadline"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "large_only_teacher_qualified_pairs": sum(p["large_only_given_teacher_and_timing"] for p in pairs),
        "small_only_teacher_qualified_pairs": sum(p["small_only_given_teacher_and_timing"] for p in pairs)}
    groups = {}
    for row in rows:
        key = row["context"]+"/"+row["controller"]
        group = groups.setdefault(key, {"n": 0, "control": 0, "sustained": 0, "joint": 0, "timing_invalid": 0})
        group["n"] += 1
        group["control"] += int(row["control_success"])
        group["sustained"] += int(row["sustained_control"])
        group["joint"] += int(row["sustained_joint"])
        group["timing_invalid"] += int(not row["timing_comparable"])
    provenance = {"driver_sha256": sha256(Path(__file__)), "setup_sha256": sha256(RESULTS/"setup.json"),
        "summary_sha256": sha256(RESULTS/"summary.json"), "raw_data": str(RESULTS),
        "destination": "internal research report; publisher requirements not selected",
        "RMSE": "error norms on0-3s, common20ms grid, trapezoidal mean-square; partial windows are missing",
        "tail": "original bracketing observations; exact2.5-3s interpolation reported separately, no relabeling",
        "replicates": "two matched repeats per recovery cell; three reference contexts but only two routes/environments; no confidence/reliability claim",
        "color": "categorical with explicit k/n; timing-incomparable counts retained as T annotations",
        "test_read": False, "training_performed": False}
    output = RESULTS/"analysis_v1"
    output.mkdir(exist_ok=True)
    atomic_json(output/"reference_control_diagnostic.json", reference_control_diagnostic(plan))
    atomic_json(output/"initial_alignment.json", initial_alignment(plan))
    atomic_json(output/"analysis.json", {"provenance": provenance, "totals": totals, "by_context_mode": groups,
        "paired_outcomes": pairs, "by_case": data["by_case"], "trials": rows})
    with (output/"trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    skill = Path("C:/Users/chen/.agents/skills/scientific-visualization")
    sys.path.insert(0, str(skill/"scripts"))
    from figure_export import export_figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    with plt.style.context(str(skill/"assets/publication.mplstyle")), plt.rc_context({
            "font.family": "DejaVu Sans", "font.size": 11, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), layout="constrained", sharey=True)
        labels = [f"{profile.replace('_', ' ')} / {period:g} ms"
                  for profile in ("baseline", "roll_minus30") for period in plan["periods_ms"]]
        for ax, context in zip(axes, plan["contexts"]):
            context_cases = [c for c in plan["cases"] if c["context"] == context["id"] and c["kind"] == "recovery"]
            for y, case in enumerate(context_cases):
                for x, controller in enumerate(CONTROLLERS):
                    count = data["by_case"][case["id"]][controller]
                    k, n, invalid = count["sustained"], count["completed"], count["timing_invalid"]
                    fill = "#0072B2" if k == n else "#F0E442" if k else "white"
                    ax.add_patch(Rectangle((x-.48, y-.46), .96, .92, facecolor=fill, edgecolor="black", linewidth=.7))
                    ax.text(x, y, f"{k}/{n}"+(f"\nT={invalid}" if invalid else ""), ha="center", va="center",
                            color="white" if k == n else "black", fontsize=11)
            nominal = next(r for r in rows if r["context"] == context["id"] and r["kind"] == "nominal")
            ax.set_title(f"env{context['environment']} route{context['route']} / phase{context['phase_s']:.2f}s\n"
                         f"Nominal teacher@50ms: {int(nominal['sustained_control'])}/1", fontsize=11)
            ax.set(xlim=(-.5, 2.5), ylim=(5.5, -.5), xticks=range(3),
                xticklabels=["QP\nteacher", "Small\nd1 w128", "Maximum\nd33 w1024"],
                yticks=range(6), yticklabels=labels)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle("Validation-context extension: sustained-control successes / completed trials", fontsize=14)
        fig.supxlabel("Blue: all; yellow: mixed; white: none. T: timing-incomparable trials (not hidden).\n"
                      "Same models and criteria; two repeats/cell, not a safety certificate.", fontsize=10)
        export_figure(fig, output/"generalization_outcomes", formats=("png", "svg"), dpi=220,
            bbox_inches=None, facecolor="white", overwrite=True, write_manifest=True, provenance=provenance)
        plt.close(fig)
    print(totals)
    print(f"Analysis: {output/'analysis.json'}")


if __name__ == "__main__":
    main()
