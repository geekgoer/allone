"""Offline validation for the MPC-complete 300-D feature contract.

This script needs neither AirSim nor ACADO by default.  Use ``--teacher`` on a
machine where the generated ACADO DLL is available to also test deterministic
label queries and repeatability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airsim_interface import RawDroneState
from falcon_paper_fidelity.mpc_complete_300d import (
    INPUT_CONTRACT,
    INPUT_DIMENSIONS,
    DeterministicAcadoTeacher,
    DeterministicTeacherConfig,
    build_mpc_complete_300d,
)
from mpc_controller import MPCReference


def make_problem() -> tuple[RawDroneState, list[MPCReference]]:
    state = RawDroneState(
        position=np.array([1.0, -2.0, -3.0]),
        quaternion=np.array([0.99, 0.01, -0.02, 0.10]),
        velocity=np.array([8.0, 0.3, -0.2]),
        acceleration=np.array([0.2, -0.1, 0.3]),
        angular_velocity=np.zeros(3),
    )
    references: list[MPCReference] = []
    for index in range(21):
        t = index * 0.1
        references.append(MPCReference(
            position=np.array([1.0 + 10.0 * t, -2.0 + 0.1 * t * t, -3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.array([10.0, 0.2 * t, 0.0]),
            acceleration=np.array([0.0, 0.2, 0.0]),
            body_rates=np.array([0.0, 0.0, 0.02]),
        ))
    return state, references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", action="store_true", help="also invoke the local ACADO DLL")
    parser.add_argument(
        "--max-refinement-steps", "--refinement-steps",
        dest="max_refinement_steps", type=int, default=16,
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    state, references = make_problem()
    feature = build_mpc_complete_300d(state, references)
    sign_flipped = [MPCReference(
        item.position, orientation=-item.orientation, velocity=item.velocity,
        acceleration=item.acceleration, body_rates=item.body_rates,
    ) for item in references]
    sign_invariant_error = float(np.max(np.abs(feature - build_mpc_complete_300d(state, sign_flipped))))
    result: dict[str, object] = {
        "status": "PASS" if feature.shape == (INPUT_DIMENSIONS,) and sign_invariant_error == 0.0 else "FAIL",
        "input_contract": INPUT_CONTRACT,
        "shape": list(feature.shape),
        "finite": bool(np.isfinite(feature).all()),
        "quaternion_sign_invariant_max_abs": sign_invariant_error,
        "first_reference_control": feature[-80:-76].tolist(),
    }
    if args.teacher:
        teacher = DeterministicAcadoTeacher(
            teacher_config=DeterministicTeacherConfig(
                maximum_refinement_steps=args.max_refinement_steps
            )
        )
        query = teacher.query(state, references)
        repeatability = teacher.reproducibility_check(state, references)
        teacher_pass = bool(query.converged and repeatability <= 1e-5)
        result["teacher"] = {
            "control": query.control.tolist(),
            "iterations": query.iterations,
            "final_control_delta": query.final_control_delta,
            "converged": query.converged,
            "diagnostics": dict(query.diagnostics),
            "repeatability_max_abs": repeatability,
            "pass": teacher_pass,
        }
        result["status"] = "PASS" if result["status"] == "PASS" and teacher_pass else "FAIL"
    text = json.dumps(result, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
        print(f"Report: {args.report.resolve()}")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
