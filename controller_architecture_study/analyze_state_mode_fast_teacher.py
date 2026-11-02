"""Nine nominal-control observations: original T50 and two new T25 repeats/context."""

import csv
from pathlib import Path

from .decision_dataset import atomic_json, sha256
from .state_mode_probe import read_json
from .state_mode_fast_teacher import PARENT, RESULTS
from .analyze_state_mode_boundary import trajectory_metrics, exact_tail_sensitivity


def main():
    if read_json(RESULTS/"summary.json")["completed"] != 6:
        raise ValueError("finish the six fixed diagnostics")
    rows = []
    for root in (PARENT, RESULTS):
        setup, data = read_json(root/"setup.json"), read_json(root/"summary.json")
        cases = {c["id"]: c for c in setup["plan"]["cases"]}
        for item in data["trials"]:
            report, tail = item["report"], item["supplementary"]
            trial = report["trial"]
            case = cases[trial["case"]]
            if case["kind"] != "nominal":
                continue
            trace = read_json(root/trial["id"]/"trace.json")
            commands = read_json(root/trial["id"]/"commands.json")
            gaps = [c["command_send_gap_ms"] for c in commands if c["slot"] > 0 and "command_send_gap_ms" in c]
            rows.append({"trial_id": trial["id"], "context": case["context"], "period_ms": case["period_ms"],
                "repeat": trial["repeat"], "control_success": report["control_success"],
                "sustained_control": tail["sustained_control_passed"], "timing_comparable": report["timing_comparable"],
                "tail_covered": tail["tail_covered"],
                "tail_max_position_error_m": tail["tail_max_position_error_m"] if tail["tail_covered"] else None,
                "tail_max_velocity_error_mps": tail["tail_max_velocity_error_mps"] if tail["tail_covered"] else None,
                "mean_command_send_gap_ms": sum(gaps)/len(gaps) if gaps else None,
                "deadline_misses": report["deadline_misses"], "delivery_deviations": report["delivery_deviations"],
                "rpc_returns_after_deadline": report["rpc_returns_after_deadline"], "error": report["error"],
                **trajectory_metrics(trace), **exact_tail_sensitivity(trace)})
    output = RESULTS/"analysis_v1"
    output.mkdir(exist_ok=True)
    atomic_json(output/"analysis.json", {"trials": rows,
        "driver_sha256": sha256(Path(__file__)), "parent_setup_sha256": sha256(PARENT/"setup.json"),
        "parent_summary_sha256": sha256(PARENT/"summary.json"), "setup_sha256": sha256(RESULTS/"setup.json"),
        "summary_sha256": sha256(RESULTS/"summary.json"), "test_read": False,
        "scope": "T25 diagnostics retain20ms target; only nominal teacher, no zero-delay test or neural-controller quality claim"})
    with (output/"trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        tail_text = (f"tail_p={row['tail_max_position_error_m']:.3f} tail_v={row['tail_max_velocity_error_mps']:.3f}"
                     if row["tail_covered"] else "tail not covered; partial trace retained")
        print(f"{row['context']} T{row['period_ms']:g}: sustained={row['sustained_control']} "
              f"timing={row['timing_comparable']} {tail_text}")
    print(f"Diagnostic analysis: {output/'analysis.json'}")


if __name__ == "__main__":
    main()
