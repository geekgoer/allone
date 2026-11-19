"""Run every historical 12-D DNN closed-loop on frozen AirSim scenarios."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_utils import load_config, resolve_project_path, select_scenarios


def _read_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    scenarios = select_scenarios(config, args.scenario_set)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = config["gate"]
    timing = config["timing"]
    speed = float(config["mpc"]["reference_speed"])
    runner = PROJECT_ROOT / "run_single_drone_dnn_gates.py"
    initial_state_root = resolve_project_path(config["initial_states"]).parent
    rows: list[dict[str, Any]] = []
    for model_item in config["models"]:
        name = str(model_item["name"])
        for scenario in scenarios:
            scenario_id = str(scenario["id"])
            report = output_dir / f"{name}_{scenario_id}.json"
            command = [
                sys.executable,
                str(runner),
                "--model", str(resolve_project_path(model_item["model"])),
                "--normalization", str(resolve_project_path(model_item["normalization"])),
                "--input-target", "gate",
                "--gate-attitude-encoding", "rotation_vector",
                "--layout-root", str(initial_state_root),
                "--initial-state-id", str(int(scenario["initial_state_id"])),
                "--device", args.device,
                "--epsilon", str(float(scenario["epsilon"])),
                "--seed", str(int(scenario["seed"])),
                "--speed", str(speed),
                "--per-gate-timeout", str(float(timing["per_gate_timeout_seconds"])),
                "--gate-count", str(int(gate["count"])),
                "--x-start", str(float(gate["x_start"])),
                "--x-spacing", str(float(gate["x_spacing"])),
                "--base-y", str(float(gate["base_y"])),
                "--base-z", str(float(gate["base_z"])),
                "--gate-width", str(float(gate["width"])),
                "--gate-height", str(float(gate["height"])),
                "--gate-thickness", str(float(gate["thickness"])),
                "--gate-clearance", str(float(gate["clearance"])),
                "--approach-distance", str(float(gate["approach_distance"])),
                "--exit-distance", str(float(gate["exit_distance"])),
                "--gate-asset", str(gate["asset"]),
                "--report", str(report),
            ]
            if args.keep_gates:
                command.append("--keep-gates")
            print(
                f"\n=== Closed loop {name} / {scenario_id} "
                f"(epsilon={scenario['epsilon']}, seed={scenario['seed']}, "
                f"initial={scenario['initial_state_id']}) ===",
                flush=True,
            )
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            report_payload = _read_report(report)
            success = completed.returncode == 0 and report_payload is not None
            row = {
                "model": name,
                "scenario_id": scenario_id,
                "epsilon": float(scenario["epsilon"]),
                "seed": int(scenario["seed"]),
                "initial_state_id": int(scenario["initial_state_id"]),
                "success": bool(success),
                "exit_code": int(completed.returncode),
                "report": str(report),
                "samples": None,
                "deadline_misses": None,
                "elapsed_seconds": None,
                "crossings": 0,
            }
            if report_payload is not None:
                row["samples"] = report_payload.get("samples")
                row["deadline_misses"] = report_payload.get("deadline_misses")
                row["elapsed_seconds"] = report_payload.get("elapsed_seconds")
                arrivals = report_payload.get("crossings", [])
                row["crossings"] = len(arrivals) if isinstance(arrivals, list) else None
            rows.append(row)
            print(
                f"RESULT {name}/{scenario_id}: "
                f"{'PASS' if success else 'FAIL'} exit_code={completed.returncode}",
                flush=True,
            )
    per_model: list[dict[str, Any]] = []
    for model_item in config["models"]:
        name = str(model_item["name"])
        selected = [row for row in rows if row["model"] == name]
        successes = sum(int(row["success"]) for row in selected)
        per_model.append({
            "model": name,
            "episodes": len(selected),
            "successes": successes,
            "failures": len(selected) - successes,
            "success_rate": successes / len(selected) if selected else None,
        })
    payload = {
        "format_version": 1,
        "benchmark": config["benchmark_name"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_set": args.scenario_set,
        "results": rows,
        "models": per_model,
        "important_note": "Closed-loop success and common-state teacher MSE are complementary metrics; neither should be silently substituted for the other.",
    }
    json_path = output_dir / "closed_loop_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "closed_loop_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()) if rows else ("model",))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {json_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "benchmark_config.json")
    parser.add_argument("--scenario-set", default="pilot", help="pilot, full, or comma-separated scenario ids")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "closed_loop_pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--keep-gates", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
