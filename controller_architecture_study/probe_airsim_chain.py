"""Two user-run, manual-Play shadow flights comparing CPU preprocessing."""

import argparse
from pathlib import Path
import subprocess
import sys

from .decision_dataset import atomic_json
from .dynamic_probe_common import new_output, read_json


STAGES = ("state_rpc_ms", "collision_rpc_ms", "preprocess_wait_ms", "reference_ms", "feature_ms",
          "mpc_solve_ms", "command_submit_ms", "queue_ms", "service_ms",
          "sample_to_mpc_dispatch_ms", "sample_to_output_ms", "release_to_output_ms")


def compact_result(report, exit_code):
    timing = report["timing"]
    return {"preprocess_policy": report["preprocess_policy"], "report": report["report_path"],
            "exit_code": exit_code, "status": report["status"], "errors": report["errors"],
            "flight_passed": report["flight_passed"], "realtime_passed": report["realtime_passed"],
            "acceptance_passed": report["acceptance_passed"], "planned_releases": report["planned_releases"],
            **{key: timing[key] for key in ("released", "ready", "skipped_release", "dropped_busy",
                                           "output_deadline_misses", "mpc_dispatch_deadline_misses")},
            "stages": {key: timing.get(key) for key in STAGES}, "flight": report["flight"]}


def run(device):
    path = new_output("airsim_chain")
    summary = {"status": "INCOMPLETE", "device": device, "model": "residual21x1024",
               "mode": "shadow", "phase": "sync", "environment_id": 48, "route_index": 0,
               "period_ms": 50., "deadline_ms": 50., "planned_trials": 2, "trials": [],
               "scope": "MPC-controlled flights with concurrent DNN inference; no DNN takeover",
               "limitations": ["Two pilot flights; no statistical or worst-case guarantee.",
                               "Flight states may differ; this is not a per-state paired accuracy test."]}
    try:
        for policy in ("parallel", "serialized"):
            output = path / policy
            summary["current_trial"] = {"preprocess_policy": policy, "output_dir": str(output)}
            atomic_json(path / "summary.json", summary)
            command = [sys.executable, "-m", "controller_architecture_study.probe_airsim_three",
                       "--mode", "shadow", "--model", summary["model"], "--device", device,
                       "--phase", "sync", "--preprocess-policy", policy,
                       "--environment-id", "48", "--route-index", "0",
                       "--period-ms", "50", "--deadline-ms", "50", "--output-dir", str(output)]
            print(f"Shadow comparison: {policy}. Use your existing manual-Play AirSim session.", flush=True)
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
            report_path = output / "report.json"
            if report_path.is_file():
                report = read_json(report_path)
                summary["trials"].append(compact_result(report, result.returncode))
            atomic_json(path / "summary.json", summary)
            if result.returncode not in (0, 2) or not report_path.is_file():
                summary["error"] = f"Flight/setup failed (exit {result.returncode}); remaining trial not started."
                return 1
        summary["status"] = "COMPLETE"
        summary["acceptance_passed"] = all(row["acceptance_passed"] for row in summary["trials"])
        print("policy      flight  realtime  ready/planned  output_misses  mpc_dispatch_misses", flush=True)
        for row in summary["trials"]:
            print(f"{row['preprocess_policy']:11s} {str(row['flight_passed']):7s} "
                  f"{str(row['realtime_passed']):9s} {row['ready']:4d}/{row['planned_releases']:<4d} "
                  f"{row['output_deadline_misses']:13d} {row['mpc_dispatch_deadline_misses']:20d}", flush=True)
        print("stage                         parallel mean/p99 ms    serialized mean/p99 ms", flush=True)
        for key in STAGES:
            values = [row["stages"][key] for row in summary["trials"]]
            cells = ["n/a" if v is None else f"{v['mean']:.3f}/{v['p99']:.3f}" for v in values]
            print(f"{key:29s} {cells[0]:>20s} {cells[1]:>25s}", flush=True)
        if not summary["acceptance_passed"]:
            print("Comparison completed; inspect timing before DNN takeover.", flush=True)
        return 0 if summary["acceptance_passed"] else 2
    finally:
        atomic_json(path / "summary.json", summary)
        print(f"Summary: {path / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    raise SystemExit(run(args.device))


if __name__ == "__main__":
    main()
