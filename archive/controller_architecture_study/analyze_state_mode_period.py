"""Descriptive outcome grid and actuator-proxy timing for the fixed-period pilot."""

import csv
from pathlib import Path
import sys

from .decision_dataset import atomic_json, sha256
from .state_mode_probe import read_json
from .state_mode_boundary import CONTROLLERS
from .state_mode_period_boundary import RESULTS
from .analyze_state_mode_boundary import trajectory_metrics, paired_outcomes, exact_tail_sensitivity


def main():
    setup = read_json(RESULTS/"setup.json")
    data = read_json(RESULTS/"summary.json")
    if data["completed"] != len(setup["plan"]["trials"]):
        raise ValueError("finish the declared period grid before final analysis")
    cases = {c["id"]: c for c in setup["plan"]["cases"]}
    rows = []
    for item in data["trials"]:
        report, tail = item["report"], item["supplementary"]
        trial = report["trial"]
        case = cases[trial["case"]]
        directory = RESULTS/trial["id"]
        commands = read_json(directory/"commands.json")
        trace = read_json(directory/"trace.json")
        gaps = [r["command_send_gap_ms"] for r in commands if "command_send_gap_ms" in r and r["slot"] > 0]
        rows.append({"trial_id": trial["id"], "case": case["source_case"], "period_ms": case["period_ms"],
            "controller": trial["mode"], "repeat": trial["repeat"],
            "control_success": report["control_success"], "sustained_control": tail["sustained_control_passed"],
            "timing_comparable": report["timing_comparable"], "sustained_joint": tail["sustained_joint_passed"],
            "tail_max_position_error_m": tail["tail_max_position_error_m"],
            "tail_max_velocity_error_mps": tail["tail_max_velocity_error_mps"],
            "minimum_h_m": report["minimum_h_m"], "progress_ratio": report["progress_ratio"],
            "recovery_confirmation_s": report["recovery_confirmation_s"],
            "max_command_send_gap_ms": max(gaps, default=None),
            "mean_command_send_gap_ms": sum(gaps)/len(gaps) if gaps else None,
            "deadline_misses": report["deadline_misses"], "delivery_deviations": report["delivery_deviations"],
            "rpc_returns_after_deadline": report["rpc_returns_after_deadline"],
            "error_kind": report["error_kind"], "error": report["error"], **trajectory_metrics(trace),
            **exact_tail_sensitivity(trace)})
    output = RESULTS/"analysis_v1"
    output.mkdir(exist_ok=True)
    provenance = {"analysis_driver_sha256": sha256(Path(__file__)),
        "setup_sha256": sha256(RESULTS/"setup.json"), "summary_sha256": sha256(RESULTS/"summary.json"),
        "raw_data": str(RESULTS), "RMSE": "20ms common grid, trapezoidal mean-square of error norm on fixed0-3s; early-ended trials missing",
        "replication": "two matched repeats per period/state/controller; no population reliability or causal mechanism certification",
        "command_gap": "host send-time proxy; first command gap excluded, not exact actuator acknowledgements",
        "tail_sensitivity": "supplementary only: linearly interpolate endpoint errors at exactly2.5/3s; do not relabel original bracketing-observation criterion",
        "test_read": False}
    analysis = {"provenance": provenance, "by_case": data["by_case"],
        "paired_outcomes": paired_outcomes(data), "trials": rows,
        "totals": {"trials": len(rows), "sustained_control": sum(r["sustained_control"] for r in rows),
            "sustained_joint": sum(r["sustained_joint"] for r in rows),
            "timing_invalid": sum(not r["timing_comparable"] for r in rows),
            "deadline_misses": sum(r["deadline_misses"] for r in rows)}}
    atomic_json(output/"analysis.json", analysis)
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
            "font.family": "DejaVu Sans", "font.size": 11, "xtick.labelsize": 11, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(8., 6.5), layout="constrained")
        for y, case in enumerate(setup["plan"]["cases"]):
            for x, controller in enumerate(CONTROLLERS):
                count = data["by_case"][case["id"]][controller]
                k, n, t = count["sustained"], count["completed"], count["timing_invalid"]
                fill = "#0072B2" if k == n else "#F0E442" if k else "white"
                text = f"{k}/{n}" + (f"\nT={t}" if t else "")
                ax.add_patch(Rectangle((x-.48, y-.46), .96, .92, facecolor=fill,
                                       edgecolor="black", linewidth=.7))
                ax.text(x, y, text, color="white" if k == n else "black", ha="center", va="center", fontsize=11)
        labels = [f"{c['source_case'].replace('_',' ')} / {c['period_ms']:g} ms" for c in setup["plan"]["cases"]]
        ax.set(xlim=(-.5, 2.5), ylim=(len(labels)-.5, -.5), xticks=range(3),
               xticklabels=["QP + attitude\nteacher", "Small\nd1 w128", "Maximum\nd33 w1024"],
               yticks=range(len(labels)), yticklabels=labels)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.suptitle("Fixed-period pilot: sustained-control successes / completed trials", fontsize=13)
        fig.supxlabel("Blue: all passes; yellow: mixed; white: none. T: timing-incomparable trials.\n"
                      "Counts are not safety certificates; first send target fixed at20ms.", fontsize=10)
        export_figure(fig, output/"period_outcomes", formats=("png", "svg"), dpi=220,
            bbox_inches=None, facecolor="white", write_manifest=True, overwrite=True,
            provenance={**provenance, "destination": "internal report; publisher requirements not selected",
                "encoding": "categorical fills plus explicit counts; all trial rows retained in trials.csv"})
        plt.close(fig)
    print(analysis["totals"])
    print(f"Analysis: {output/'analysis.json'}")


if __name__ == "__main__":
    main()
