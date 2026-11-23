"""Freeze the two observed slow-ACK cases and each fixed/retimed configuration."""

import argparse
import json
from pathlib import Path
import shutil

from .critical_update_screen import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.capacity_inputs / "config_n10_t20.json")
    if (len(base["names"]) != 10 or base["period_ms"] != 20 or base["model_id"] != 11 or
            not base["single_command_check"] or not base["isolate_command_costs"] or base["adaptive"]):
        parser.error("Use the frozen ten-vehicle 20 ms capacity arm")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.capacity_inputs / "case.json", args.output / "case.json")
    shutil.copyfile(args.capacity_inputs / "vehicles_n10_t20.txt", args.output / "vehicles.txt")
    for vehicle, cycle, delay in (("Drone2", 1139, 43.206), ("Drone5", 568, 45.320)):
        if vehicle not in base["names"]:
            parser.error(f"Missing probe vehicle {vehicle}")
        for label, enabled in (("off", False), ("on", True)):
            probe = dict(vehicle=vehicle, cycle=cycle, ack_delay_ms=delay, retime=enabled)
            config = dict(base, command_retime_probe=probe)
            (args.output / f"config_{vehicle.lower()}_{label}.json").write_text(
                json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
