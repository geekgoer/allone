"""Validate the public FALCON protocol before collection or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "paper_core_config.json")
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checks = {
        "initial_states": int(config["initial_state"]["count"]) == 10,
        "environments": int(config["environment"]["count"]) == 40,
        "dt_50ms": bool(np.isclose(float(config["timing"]["dt"]), 0.05)),
        "episode_5s": bool(np.isclose(float(config["timing"]["episode_seconds"]), 5.0)),
        "steps_100": int(config["timing"]["steps_per_episode"]) == 100,
        "horizon_20": int(config["mpc"]["horizon"]) == 20,
        "five_gates": int(config["environment"]["gate_count"]) == 5,
        "fixed_planner_speed": float(config["planner"]["nominal_speed"]) == float(config["planner"]["terminal_speed"]),
    }
    report: dict[str, object] = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_samples": 10 * 40 * 100,
        "note": "5 m/s and unreleased polynomial/solver details are explicit engineering assumptions",
    }
    if args.dataset is not None:
        with np.load(args.dataset, allow_pickle=False) as archive:
            report["X_shape"] = list(archive["X"].shape)
            report["U_shape"] = list(archive["U"].shape)
            report["finite"] = bool(np.isfinite(archive["X"]).all() and np.isfinite(archive["U"]).all())
        if report["X_shape"] != [40000, 12] or report["U_shape"] != [40000, 4] or not report["finite"]:
            report["status"] = "FAIL"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
