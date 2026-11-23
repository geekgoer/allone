"""Find turn-window releases that an ordering change could actually reach."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


WINDOWS = {"early": (0.80, 1.10), "late": (0.075, 0.225)}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(folder):
    with (folder / "requests.jsonl").open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def stage(row):
    if row["status"] == "delivered":
        return "delivered"
    if row.get("inference_dispatched"):
        return "miss_after_dispatch"
    queued = row.get("calendar_queued", {})
    if "dpu_enqueue_us" in row and not queued.get("cancelled", True):
        return "miss_while_ready"
    return "miss_before_ready"


def screen(trials):
    counts = defaultdict(Counter)
    occurrences = defaultdict(Counter)
    source = None
    for folder in trials:
        identity = load(folder / "execution_identity.json")
        prepared = load(folder / "prepared_source.json")
        signature = (prepared["original_case"], prepared["vehicles"], identity["period_ms"],
                     prepared["reference_start_s"], identity["duration_s"])
        if source is not None and source != signature:
            raise ValueError("Screen different cases, vehicles or release calendars separately")
        source = signature
        case, _, period, start, _ = signature
        for row in rows(folder):
            if row["period_ms"] != period:
                raise ValueError("Screen requires a fixed period in the recorded release rows")
            t_ref = start + (row["planned_release_us"] - identity["origin_us"]) / 1e6
            vehicle = row["vehicle"]
            for gate in case["layouts"][vehicle]["gates"]:
                if gate.get("kind") != "turn":
                    continue
                lead = gate["reference_time_s"] - t_ref
                for window, (lower, upper) in WINDOWS.items():
                    if lower <= lead <= upper:
                        result = stage(row)
                        counts[window][result] += 1
                        occurrences[(vehicle, gate["index"], row["cycle"], window)][result] += 1
    candidates = []
    for (vehicle, gate, cycle, window), seen in occurrences.items():
        if seen["delivered"] and seen["miss_while_ready"]:
            candidates.append(dict(vehicle=vehicle, gate=gate, cycle=cycle, window=window, stages=seen))
    candidates.sort(key=lambda x: (x["window"], -x["stages"]["miss_while_ready"],
                                   -x["stages"]["delivered"], x["vehicle"], x["gate"], x["cycle"]))
    return dict(exploratory=True, runs=[str(path) for path in trials], windows_before_reference_gate_s=WINDOWS,
                period_ms=source[2], counts=counts, candidates=candidates,
                scope="Queue exposure and release alignment only. No physical importance or order repair proven.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen(args.trials)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"counts": result["counts"], "candidate_count": len(result["candidates"])}, indent=2))


if __name__ == "__main__":
    main()
