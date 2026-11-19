"""Immutable-input numerical tables and descriptive figures for the boundary pilot."""

import csv
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .state_mode_boundary import CONTROLLERS, RESULTS
from .state_mode_probe import read_json


def trajectory_metrics(trace, duration=3.):
    if not trace or trace[0]["t"] > 0. or trace[-1]["t"] < duration:
        return {"full_window_rmse_available": False, "position_rmse_m": None, "velocity_rmse_mps": None}
    times = np.asarray([r["t"] for r in trace])
    if not np.all(np.diff(times) > 0.):
        raise ValueError("trace timestamps must strictly increase")
    grid = np.linspace(0., duration, 151)
    position = np.interp(grid, times, [r["position_error_m"] for r in trace])
    velocity = np.interp(grid, times, [r["velocity_error_mps"] for r in trace])
    return {"full_window_rmse_available": True,
        "position_rmse_m": float(np.sqrt(np.trapz(position**2, grid)/duration)),
        "velocity_rmse_mps": float(np.sqrt(np.trapz(velocity**2, grid)/duration))}


def exact_tail_sensitivity(trace, duration=3.):
    """Analysis-only sensitivity to bracketing observations outside [2.5,3]s."""
    start = duration-.5
    if not trace or trace[0]["t"] > start or trace[-1]["t"] < duration:
        return {"exact_tail_available": False, "exact_tail_passed": False,
            "exact_tail_max_position_m": None, "exact_tail_max_velocity_mps": None}
    times = np.asarray([r["t"] for r in trace])
    grid = np.r_[start, times[(times > start) & (times < duration)], duration]
    position = np.interp(grid, times, [r["position_error_m"] for r in trace])
    velocity = np.interp(grid, times, [r["velocity_error_mps"] for r in trace])
    return {"exact_tail_available": True, "exact_tail_passed": bool(position.max() <= .3 and velocity.max() <= .6),
        "exact_tail_max_position_m": float(position.max()), "exact_tail_max_velocity_mps": float(velocity.max())}


def result_rows(phase, data):
    rows = []
    for item in data["trials"]:
        report, tail = item["report"], item["supplementary"]
        trial = report["trial"]
        directory = RESULTS/phase/trial["id"]
        trace = read_json(directory/"trace.json")
        rows.append({"phase": phase, "trial_id": trial["id"], "case": trial["case"],
            "controller": trial["mode"], "repeat": trial["repeat"],
            "control_success": report["control_success"], "sustained_control": tail["sustained_control_passed"],
            "timing_comparable": report["timing_comparable"], "sustained_joint": tail["sustained_joint_passed"],
            "minimum_h_m": report["minimum_h_m"], "progress_ratio": report["progress_ratio"],
            "recovery_confirmation_s": report["recovery_confirmation_s"],
            "tail_max_position_error_m": tail["tail_max_position_error_m"],
            "tail_max_velocity_error_mps": tail["tail_max_velocity_error_mps"],
            "deadline_misses": report["deadline_misses"], "delivery_deviations": report["delivery_deviations"],
            "rpc_returns_after_deadline": report["rpc_returns_after_deadline"],
            "error_kind": report["error_kind"], "error": report["error"], **trajectory_metrics(trace),
            **exact_tail_sensitivity(trace)})
    return rows


def paired_outcomes(data):
    paired = []
    keys = sorted({(r["report"]["trial"]["case"], r["report"]["trial"]["repeat"]) for r in data["trials"]})
    for case, repeat in keys:
        found = {r["report"]["trial"]["mode"]: r for r in data["trials"]
                 if r["report"]["trial"]["case"] == case and r["report"]["trial"]["repeat"] == repeat}
        complete = set(found) == set(CONTROLLERS)
        comparable = complete and all(r["report"]["timing_comparable"] for r in found.values())
        teacher = complete and found[CONTROLLERS[0]]["supplementary"]["sustained_control_passed"]
        small = complete and found[CONTROLLERS[1]]["supplementary"]["sustained_control_passed"]
        large = complete and found[CONTROLLERS[2]]["supplementary"]["sustained_control_passed"]
        paired.append({"case": case, "repeat": repeat, "all_timings_comparable": bool(comparable),
            "teacher_sustained": bool(teacher), "small_sustained": bool(small), "large_sustained": bool(large),
            "large_only_with_comparable_timing": bool(comparable and large and not small),
            "small_only_with_comparable_timing": bool(comparable and small and not large),
            "large_only_given_teacher_and_timing": bool(comparable and teacher and large and not small),
            "small_only_given_teacher_and_timing": bool(comparable and teacher and small and not large)})
    return paired


def export_observed_evidence(output):
    """Observed full-state evidence only; unmeasured/mixed cells stay unknown."""
    root_cases = {c["id"]: c for c in read_json(RESULTS/"setup.json")["cases"]}
    states = ("baseline", "outward_v2", "roll_minus30")
    periods = (50., 75., 100., 125., 150., 200.)
    sources = [(RESULTS/"discovery", 50.), (RESULTS/"confirmation", 50.),
        (RESULTS.with_name("state_mode_period_boundary_v1"), None),
        (RESULTS.with_name("state_mode_period_refinement_v1"), None)]
    grouped, source_log = {}, []
    for directory, fixed_period in sources:
        path = directory/"summary.json"
        if not path.exists():
            source_log.append({"path": str(path), "used": False, "reason": "not present"})
            continue
        data, setup = read_json(path), read_json(directory/"setup.json")
        plan = setup if fixed_period is not None else setup["plan"]
        if data["completed"] != len(plan["trials"]):
            source_log.append({"path": str(path), "used": False, "reason": "incomplete; not silently pooled"})
            continue
        cases = root_cases if fixed_period is not None else {c["id"]: c for c in plan["cases"]}
        source_log.append({"path": str(path), "used": True, "sha256": sha256(path)})
        for item in data["trials"]:
            report, trial = item["report"], item["report"]["trial"]
            case = cases[trial["case"]]
            state = case.get("source_case", case["id"])
            if state not in states:
                continue
            for key in ("position", "velocity", "quaternion", "held_control", "reference_phase_s"):
                if not np.allclose(case[key], root_cases[state][key], rtol=0., atol=1e-12):
                    raise ValueError(f"different physical/reference context cannot be pooled: {state}/{key}")
            if case["route_sha256"] != root_cases[state]["route_sha256"]:
                raise ValueError("route provenance differs")
            period = fixed_period if fixed_period is not None else case["period_ms"]
            grouped.setdefault((state, period, trial["mode"]), []).append((directory.name, item))
    entries = []
    for state in states:
        for period in periods:
            for mode in CONTROLLERS:
                found = grouped.get((state, period, mode), [])
                n = len(found)
                k = sum(r["supplementary"]["sustained_control_passed"] for _, r in found)
                invalid = sum(not r["report"]["timing_comparable"] for _, r in found)
                status = ("not_tested" if not n else "mixed_or_timing_unknown" if invalid or 0 < k < n
                          else "observed_all_passed" if k == n else "observed_all_failed")
                entries.append({"case": state, "period_ms": period, "controller": mode, "n": n,
                    "sustained": k, "timing_invalid": invalid, "status": status,
                    "three_repeat_pilot_eligible": bool(n >= 3 and k == n and not invalid),
                    "source_trials": [{"phase": phase, "trial_id": r["report"]["trial"]["id"]} for phase, r in found]})
    aliases = []
    for a, b in combinations(states, 2):
        if any(root_cases[a][k] != root_cases[b][k] for k in ("requested_h_m", "requested_h_dot_mps")):
            continue
        for period in periods:
            for mode in CONTROLLERS:
                pair = [next(e for e in entries if e["case"] == c and e["period_ms"] == period
                             and e["controller"] == mode) for c in (a, b)]
                if all(e["n"] >= 3 and not e["timing_invalid"] for e in pair) and {
                        e["status"] for e in pair} == {"observed_all_passed", "observed_all_failed"}:
                    aliases.append({"cases": [a, b], "h_m": root_cases[a]["requested_h_m"],
                        "h_dot_mps": root_cases[a]["requested_h_dot_mps"], "period_ms": period,
                        "controller": mode, "outcomes": [f"{e['sustained']}/{e['n']}" for e in pair]})
    evidence = {"scope": "three exact full-state/reference profiles only; NOT a general runtime lookup or safety certificate",
        "warning": "three_repeat_pilot_eligible is finite pilot evidence, not certified admissibility; untested/mixed/OOD remain unknown",
        "aggregation": "matching requested physics/reference verified; per-phase trial identifiers preserved; repeats are not independent environments",
        "sources": source_log, "entries": entries, "two_feature_alias_counterexamples": aliases,
        "analysis_driver_sha256": sha256(Path(__file__)), "test_read": False}
    atomic_json(output/"observed_evidence_matrix.json", evidence)


def make_figure(phases, output, provenance):
    # Reuse the installed scientific-visualization skill's export/style assets.
    # Numerical CSV/JSON remain independent of these presentation helpers.
    skill = Path("C:/Users/chen/.agents/skills/scientific-visualization")
    sys.path.insert(0, str(skill/"scripts"))
    from figure_export import export_figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labels = ["QP + attitude\nteacher", "Small\nd1 w128", "Maximum\nd33 w1024"]
    rc = {"font.size": 10, "xtick.labelsize": 10, "ytick.labelsize": 9,
          "axes.titlesize": 12, "font.family": "DejaVu Sans"}
    with plt.style.context(str(skill/"assets/publication.mplstyle")), plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(phases), figsize=(12., 8.), layout="constrained", squeeze=False)
        case_order = [c["id"] for c in read_json(RESULTS/"setup.json")["cases"]]
        for index, (ax, (phase, data)) in enumerate(zip(axes[0], phases.items())):
            sampled = read_json(RESULTS/phase/"setup.json")["cases"]
            for y, case in enumerate(case_order):
                for x, mode in enumerate(CONTROLLERS):
                    count = data["by_case"].get(case, {}).get(mode)
                    if not count or not count["completed"]:
                        color, text, ink = "#D0D0D0", "not\nrepeated" if case not in sampled else "pending", "black"
                    else:
                        n, k, timing = count["completed"], count["sustained"], count["timing_invalid"]
                        color = "#0072B2" if k == n else "#F0E442" if k else "white"
                        ink = "white" if k == n else "black"
                        text = f"{k}/{n}" + (f"\nT={timing}" if timing else "")
                    ax.add_patch(Rectangle((x-.48, y-.46), .96, .92, facecolor=color,
                                           edgecolor="black", linewidth=.65))
                    ax.text(x, y, text, ha="center", va="center", color=ink, fontsize=10)
            ax.set(xlim=(-.5, 2.5), ylim=(len(case_order)-.5, -.5),
                   xticks=range(3), xticklabels=labels, yticks=range(len(case_order)),
                   yticklabels=[c.replace("_", " ") for c in case_order] if index == 0 else [])
            ax.set_title(("A  Discovery: one trial per cell" if phase == "discovery"
                          else "B  Selected confirmation: three NEW trials per cell"), loc="left", pad=12)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle("Boundary pilot: observed sustained-control successes / completed trials", fontsize=14)
        fig.supxlabel("Blue: all sustained-control passes; yellow: mixed; white: none; gray: not sampled (not a failure).\n"
                      "T = timing-incomparable trials (not a control failure). No cell is a safety certificate.", fontsize=10)
        export_figure(fig, output/"boundary_outcomes", formats=("png", "svg"), dpi=220,
            bbox_inches=None, facecolor="white", write_manifest=True, overwrite=True,
            provenance={**provenance, "destination": "internal research report; no publisher-specific compliance claim",
                "replication": "discovery n=1 per cell; confirmation n=3 selected cases only; not independent population trials",
                "encoding": "categorical fills with explicit counts; timing invalidity retained as T; missing is not zero"})
        plt.close(fig)


def main():
    output = RESULTS/"analysis_v1"
    output.mkdir(exist_ok=True)
    phases, rows, pairs, inputs = {}, [], {}, {}
    for phase in ("discovery", "confirmation"):
        path = RESULTS/phase/"summary.json"
        data = read_json(path)
        plan = read_json(path.with_name("setup.json"))
        if data["completed"] != len(plan["trials"]):
            raise ValueError(f"{phase} is not complete; do not plot partial data as final")
        phases[phase] = data
        rows.extend(result_rows(phase, data))
        pairs[phase] = paired_outcomes(data)
        inputs[phase] = {"summary_sha256": sha256(path), "setup_sha256": sha256(path.with_name("setup.json"))}
    setup = read_json(RESULTS/"setup.json")
    original = read_json(ROOT/"controller_architecture_study/results/state_mode_admissibility_v2/setup.json")["plan"]
    normalization = read_json(Path(original["export_dir"])/"normalization.json")
    initial_errors = []
    for item in read_json(RESULTS/"discovery/initial_outputs.json"):
        target = np.asarray(item["outputs"][CONTROLLERS[0]])
        initial_errors.append({"case": item["case"], **{
            m: float(np.mean(((np.asarray(item["outputs"][m])-target)/normalization["u_std"])**2))
            for m in CONTROLLERS[1:]}})
    provenance = {"driver_sha256": sha256(Path(__file__)), "boundary_setup_sha256": sha256(RESULTS/"setup.json"),
        "inputs": inputs, "normalization_sha256": sha256(Path(original["export_dir"])/"normalization.json"),
        "RMSE": "sqrt(integral_0^3 norm(error(t))^2 dt / 3); linear interpolation at 20ms grid, trapezoidal integration",
        "early_termination": "full-window RMSE missing, not zero or partial-window average",
        "selection": setup["confirmation_rule"], "test_read": False}
    analysis = {"provenance": provenance, "paired_outcomes": pairs, "initial_aligned_normalized_mse": initial_errors,
        "totals": {phase: {"trials": data["completed"],
            "sustained_control": sum(r["supplementary"]["sustained_control_passed"] for r in data["trials"]),
            "sustained_joint": sum(r["supplementary"]["sustained_joint_passed"] for r in data["trials"]),
            "timing_invalid": sum(not r["report"]["timing_comparable"] for r in data["trials"]),
            "large_only_comparable_pairs": sum(r["large_only_given_teacher_and_timing"] for r in pairs[phase]),
            "small_only_comparable_pairs": sum(r["small_only_given_teacher_and_timing"] for r in pairs[phase])}
            for phase, data in phases.items()}, "trials": rows}
    atomic_json(output/"analysis.json", analysis)
    with (output/"trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_figure(phases, output, provenance)
    export_observed_evidence(output)
    print(analysis["totals"])
    print(f"Analysis: {output/'analysis.json'}")


if __name__ == "__main__":
    main()
