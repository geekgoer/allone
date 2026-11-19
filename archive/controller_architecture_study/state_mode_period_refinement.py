"""One midpoint per teacher pass/fail bracket, at most27 NEW trials; no repeated search."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from . import state_mode_period_boundary as engine
from .decision_dataset import atomic_json, sha256
from .state_mode_boundary import RESULTS as BOUNDARY, CONTROLLERS, build_cases
from .state_mode_probe import read_json, make_trials


COARSE = engine.RESULTS
RESULTS = COARSE.with_name("state_mode_period_refinement_v1")


def select_midpoints(coarse, at50):
    brackets, skipped = [], {}
    for state in engine.STATES:
        base = at50["by_case"][state]["QP_teacher"]
        if base["timing_invalid"] or base["joint"] != base["completed"]:
            skipped[state] = "50ms teacher evidence is not fully passing/comparable"
            continue
        lower, upper = 50., None
        for period in engine.PERIODS_MS:
            count = coarse["by_case"][f"{state}_T{period:g}"]["QP_teacher"]
            if count["completed"] != 2 or count["timing_invalid"]:
                skipped[state] = "coarse teacher cell incomplete or timing-incomparable"
                break
            if count["sustained"] == 2:
                lower = period
            elif count["sustained"] == 0:
                upper = period
                break
            else:
                skipped[state] = "mixed teacher outcomes; no unambiguous observed bracket"
                break
        if upper is not None:
            brackets.append({"source_case": state, "observed_passing_period_ms": lower,
                "observed_failing_period_ms": upper, "period_ms": (lower+upper)/2})
        elif state not in skipped:
            skipped[state] = "no observed teacher failure in the fixed coarse grid"
    return brackets, skipped


def build_refinement(brackets):
    original, layout = build_cases()
    cases = []
    for bracket in brackets:
        case = deepcopy(next(c for c in original if c["id"] == bracket["source_case"]))
        case.update(bracket, id=f"{bracket['source_case']}_T{bracket['period_ms']:g}")
        cases.append(case)
    trials = make_trials(cases, [{"name": m} for m in CONTROLLERS], 3, 20261026)
    return {"cases": cases, "trials": trials, "repeats": 3, "duration_s": 3.,
        "scope": "single teacher-bracket midpoint per state, three matched repeats; max27 trials; never recursively bisected",
        "refinement_driver_sha256": sha256(Path(__file__)),
        "coarse_summary_sha256": sha256(COARSE/"summary.json"),
        "selection": "uses teacher pass/fail only, not desired small/large outcomes",
        "test_read": False, "training_performed": False}, layout


def main():
    coarse, plan = read_json(COARSE/"summary.json"), read_json(COARSE/"setup.json")["plan"]
    if coarse["completed"] != len(plan["trials"]):
        raise ValueError("finish all declared coarse period trials before a single refinement")
    at50 = read_json(BOUNDARY/"discovery/summary.json")
    brackets, skipped = select_midpoints(coarse, at50)
    selection = {"rule": "one midpoint of first fully comparable teacher pass/fail bracket; max3 states,9 trials/state",
        "coarse_sha256": sha256(COARSE/"summary.json"), "at50_sha256": sha256(BOUNDARY/"discovery/summary.json"),
        "driver_sha256": sha256(Path(__file__)), "brackets": brackets, "skipped": skipped,
        "warning": "adaptive pilot selection, not an independent preregistered significance test or a certified T_max"}
    RESULTS.mkdir(exist_ok=True)
    if (RESULTS/"selection.json").exists() and read_json(RESULTS/"selection.json") != selection:
        raise ValueError("midpoint selection changed; preserve previous version")
    atomic_json(RESULTS/"selection.json", selection)
    if not brackets:
        print(f"No eligible teacher bracket; no extra flights. Selection: {RESULTS/'selection.json'}")
        return
    print(f"Single-step midpoint refinement: {brackets}", flush=True)
    # Parameterize the frozen experiment engine without editing it. The new
    # plan records this wrapper's hash as well as the unchanged engine hash.
    with patch.object(engine, "RESULTS", RESULTS), \
         patch.object(engine, "PERIODS_MS", tuple(sorted({b["period_ms"] for b in brackets}))), \
         patch.object(engine, "build_plan", lambda: build_refinement(brackets)):
        engine.main()
    from . import analyze_state_mode_period as analyze
    with patch.object(analyze, "RESULTS", RESULTS):
        analyze.main()


if __name__ == "__main__":
    main()
