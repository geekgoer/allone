"""Time-scale the frozen heterogeneous path without changing its geometry."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from controller_architecture_study.ip37_safety_cases import audit


def scale_case(source, rate):
    case = deepcopy(source)
    for name, layout in case["layouts"].items():
        for segment in layout["segments"]:
            segment["start_time"] /= rate
            segment["duration"] /= rate
            segment["coefficients"] = [
                [value * rate ** order for order, value in enumerate(axis)]
                for axis in segment["coefficients"]
            ]
        for gate in layout["gates"]:
            gate["reference_time_s"] /= rate
            if "reference_velocity" in gate:
                gate["reference_velocity"] = [value * rate for value in gate["reference_velocity"]]
            if "reference_acceleration" in gate:
                gate["reference_acceleration"] = [value * rate ** 2 for value in gate["reference_acceleration"]]
        layout["duration"] /= rate
        case["roles"][name]["turn_gate_times_s"] = [t / rate for t in case["roles"][name]["turn_gate_times_s"]]
    case["duration_s"] = float(int(max(layout["duration"] for layout in case["layouts"].values()) + 3))
    case["time_scale_rate"] = rate
    case["time_scale_note"] = "p_new(t)=p_frozen(rate*t); same gate centers and apertures; no disturbance"
    case["live_flight_verified"] = False
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new directory")
    original = json.loads(args.source.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    for rate in (0.8, 0.7):
        case = scale_case(original, rate)
        report = audit(case)
        if not report["passed"]:
            raise ValueError(f"reference geometry failed at rate {rate}")
        label = f"case_w{original['net_width_m']:.1f}_r{rate:.1f}"
        (args.output / f"{label}.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
        (args.output / f"{label}_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(label, "duration_s", case["duration_s"], "speed_max", max(
            x["max_sampled_speed_mps"] for x in report["vehicles"].values()))


if __name__ == "__main__":
    main()
