"""Validate the corrected ACADO paper-core configuration and final dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from falcon_paper_fidelity.collector import ACADO_POLICY_VERSION  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "paper_core_config.json")
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    library = Path(config["mpc"]["library"])
    library = library if library.is_absolute() else (config_path.parent / library).resolve()
    checks = {
        "initial_states_10": int(config["initial_state"]["count"]) == 10,
        "environments_40": int(config["environment"]["count"]) == 40,
        "control_period_50ms": bool(np.isclose(float(config["timing"]["dt"]), 0.05)),
        "episode_5s": bool(np.isclose(float(config["timing"]["episode_seconds"]), 5.0)),
        "steps_100": int(config["timing"]["steps_per_episode"]) == 100,
        "horizon_20": int(config["mpc"]["horizon"]) == 20,
        "official_acado_backend": config["mpc"]["backend"] == "official_rpg_mpc_acado_qpoases",
        "generated_solver_prediction_dt_100ms": bool(
            np.isclose(float(config["mpc"]["prediction_dt"]), 0.1)
        ),
        "solver_dll_exists": library.is_file(),
        "five_gates": int(config["environment"]["gate_count"]) == 5,
        "fixed_reference_speed": float(config["planner"]["nominal_speed"])
        == float(config["planner"]["terminal_speed"]),
        "isolated_v2_output": config["dataset"]["root"] == "paper_core_40k_acado_v2",
    }
    report: dict[str, object] = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "teacher_policy_version": ACADO_POLICY_VERSION,
        "checks": checks,
        "expected_samples": 40000,
        "solver_library": str(library),
        "smoke_protocol": "2 initial states x 2 environments x full 100 steps",
    }
    if args.dataset is not None:
        with np.load(args.dataset, allow_pickle=False) as archive:
            report["X_shape"] = list(archive["X"].shape)
            report["U_shape"] = list(archive["U"].shape)
            report["finite"] = bool(
                np.isfinite(archive["X"]).all() and np.isfinite(archive["U"]).all()
            )
        if (
            report["X_shape"] != [40000, 12]
            or report["U_shape"] != [40000, 4]
            or not report["finite"]
        ):
            report["status"] = "FAIL"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
