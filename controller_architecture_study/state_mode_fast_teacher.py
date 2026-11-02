"""Six nominal teacher diagnostics: halve T, preserve the20ms response target."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from .decision_dataset import sha256
from .state_mode_probe import make_trials, read_json
from . import state_mode_generalization as engine


PARENT = engine.RESULTS
RESULTS = PARENT.with_name("state_mode_fast_teacher_v1")
parent_plan = engine.build_plan


def build_plan():
    plan, layouts = parent_plan()
    cases = []
    for source in plan["cases"]:
        if source["kind"] == "nominal":
            case = deepcopy(source)
            case.update(id=f"{source['context']}_nominal_T25", period_ms=25.)
            cases.append(case)
    plan.update(cases=cases, trials=make_trials(cases, [{"name": "QP_teacher"}], 2, 20261028),
        controllers=["QP_teacher"], periods_ms=[25.],
        wrapper_sha256=sha256(Path(__file__)),
        parent_setup_sha256=sha256(PARENT/"setup.json"),
        parent_summary_sha256=sha256(PARENT/"summary.json"),
        selection="All3 frozen contexts, nominal teacher only; T25 vs originalT50, unchanged20ms delivery. No outcome filtering.",
        scope="Six diagnostics of whether reducing update period alone meets the same quality criterion; "
              "not a zero-delay test, not a controller redesign or reliability certificate")
    return plan, layouts


def main():
    parent = read_json(PARENT/"summary.json")
    if parent["completed"] != parent["expected"]:
        raise ValueError("finish the full generalization grid first")
    with patch.object(engine, "RESULTS", RESULTS), patch.object(engine, "build_plan", build_plan):
        engine.main()


if __name__ == "__main__":
    main()
