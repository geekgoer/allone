"""Screen static release phases, then freeze same-model capacity comparisons."""

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

from .critical_update_screen import load, rows


COMPARE_PERIOD_MS = 20.0
MODEL_ID = 11
EXCLUDED = {"Drone6", "Drone12"}


def grid(names, period):
    return {name: i * period / len(names) for i, name in enumerate(names)}


def same_case(a, b):
    # Prepared case JSON may have been reserialized at reduced float precision.
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_case(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_case(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return type(a) in (float, int) and type(b) in (float, int) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def load_ready_delays(trial, case, names):
    identity = load(trial / "execution_identity.json")
    setup = load(trial / "prepared_source.json")
    if (not same_case(setup["original_case"], case) or sorted(setup["vehicles"]) != names or
            identity["period_ms"] != COMPARE_PERIOD_MS):
        raise ValueError(f"Source does not match the frozen 12-vehicle, 20 ms case: {trial}")
    delays = []
    for row in rows(trial):
        if row["period_ms"] != COMPARE_PERIOD_MS:
            raise ValueError(f"Only static-period traces can be replayed: {trial}")
        if "dpu_enqueue_us" not in row:
            continue
        delay = (row["dpu_enqueue_us"] - row["planned_release_us"]) / 1000
        if delay < 0:
            raise ValueError(f"Negative readiness delay: {trial}")
        delays.append((row["cycle"], row["vehicle"], delay))
    if not delays:
        raise ValueError(f"Trace had no ready jobs: {trial}")
    return delays


def replay(delays, phases, period, model_ms, reserve_ms):
    jobs = sorted((cycle * period + phases[v] + delay, cycle * period + phases[v] + period, v, cycle)
                  for cycle, v, delay in delays if v in phases and cycle * period + phases[v] < 32000)
    waiting = []
    i = 0
    t = 0.0
    misses = 0
    waits = []
    while waiting or i < len(jobs):
        if not waiting:
            t = max(t, jobs[i][0])
        while i < len(jobs) and jobs[i][0] <= t:
            ready, due, v, cycle = jobs[i]
            heapq.heappush(waiting, (due, v, cycle, ready))
            i += 1
        due, _, _, ready = heapq.heappop(waiting)
        waits.append(t - ready)
        if t + model_ms + reserve_ms >= due:
            misses += 1
        else:
            t += model_ms
    waits.sort()
    return dict(known_ready_jobs=len(waits), projected_misses=misses,
                wait_p99_ms=round(waits[(99 * len(waits) - 1) // 100], 6),
                wait_max_ms=round(waits[-1], 6))


def score(r):
    return r["projected_misses"], r["wait_p99_ms"]


def screen(names, training, validation, model_ms, reserve_ms):
    baseline = grid(names, COMPARE_PERIOD_MS)
    orders = [names] + [names[i:] + names[:i] for i in range(1, len(names))]
    orders.append(names[::-1])
    for i in range(len(names) - 1):
        alt = names.copy()
        alt[i], alt[i + 1] = alt[i + 1], alt[i]
        orders.append(alt)

    def evaluate(phases, data):
        return replay(data, phases, COMPARE_PERIOD_MS, model_ms, reserve_ms)

    candidates = []
    for order in orders:
        phases = grid(order, COMPARE_PERIOD_MS)
        train = [evaluate(phases, seq) for seq in training]
        candidates.append((phases, train))
    champion, train = min(candidates, key=lambda x: tuple(map(sum, zip(*map(score, x[1])))))
    incumbent_train = candidates[0][1]
    independent = (evaluate(baseline, validation), evaluate(champion, validation))
    # Require fewer projected misses in the held-out run, and no increase on
    # either training run, before changing the already-regular baseline.
    select = (independent[1]["projected_misses"] < independent[0]["projected_misses"] and
              all(x["projected_misses"] <= y["projected_misses"] for x, y in zip(train, incumbent_train)))
    return dict(candidate_count=len(candidates), model_ms=model_ms, reserve_ms=reserve_ms,
                uniform_phase_ms=baseline, best_training_phase_ms=champion,
                training_uniform=incumbent_train, training_best=train,
                held_out_uniform=independent[0], held_out_best=independent[1],
                selected="tuned" if select else "uniform", selected_phase_ms=champion if select else baseline)


def prepare(case_path, config_path, training_paths, validation_path, output):
    case = load(case_path)
    config = load(config_path)
    names = sorted(case["layouts"])
    if len(names) != 12 or any(case["roles"][v]["role"] != "smooth" for v in EXCLUDED):
        raise ValueError("Expected the frozen twelve-vehicle layout with two smooth removal candidates")
    pool = {mode["numeric_id"]: mode for mode in config["modes"]}
    if MODEL_ID not in config["candidate_ids"] or "initial_periods" in config:
        raise ValueError("Expected the common model pool and uniform period")
    training = [load_ready_delays(t, case, names) for t in training_paths]
    validation = load_ready_delays(validation_path, case, names)
    report = screen(names, training, validation, pool[MODEL_ID]["service_ms"], config["command_reserve_ms"])
    report.update(training=[str(t) for t in training_paths], validation=str(validation_path),
                  scope="Known-ready jobs only; fixed per-job readiness delays and fixed empirical service/reserve.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "case.json").write_bytes(case_path.read_bytes())
    shared_phases = report["selected_phase_ms"]
    for trial, roster, period in (("n12_t20", names, 20.), ("n10_t20", [v for v in names if v not in EXCLUDED], 20.),
                                  ("n12_t30", names, 30.)):
        frozen = dict(config, names=roster, period_ms=period, model_id=MODEL_ID, adaptive=False, enable_skip=False,
                      initial_phase_ms={v: shared_phases[v] for v in roster})
        (output / f"config_{trial}.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        (output / f"vehicles_{trial}.txt").write_bytes((",".join(roster) + "\n").encode("ascii"))
    (output / "phase_screen.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if "phase_ms" not in k}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "config", "validation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--training", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.validation.resolve() in [t.resolve() for t in args.training]:
        parser.error("Validation must be a different flight from every training run")
    prepare(args.case, args.config, args.training, args.validation, args.output)


if __name__ == "__main__":
    main()
