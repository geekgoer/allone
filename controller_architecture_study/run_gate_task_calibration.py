"""Recollect the mode map with the online gate-task predicate on isolated CPUs."""

import sys
from pathlib import Path
import shutil

from .decision_dataset import ROOT, atomic_json, sha256
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime_calibrate import main as calibrate
from .framework_runtime_editor import read_background_throttle


OUTPUT = ROOT/"controller_architecture_study/results/framework_gate_task_calibration_v2_direct_collision"


def main():
    sys.setswitchinterval(.0005)
    editor = read_background_throttle()
    if editor["background_throttle_enabled"]:
        raise RuntimeError("disable background throttling before comparable calibration")
    frozen = OUTPUT/"frozen_source"
    frozen.mkdir(parents=True,exist_ok=True)
    names = ("run_gate_task_calibration.py","framework_runtime_calibrate.py","framework_runtime.py",
        "framework_runtime_world.py","safety_comparison_flight.py","safety_comparison_core.py")
    for name in names:
        source,destination = Path(__file__).with_name(name),frozen/name
        if not destination.exists():
            shutil.copy2(source,destination)
        if sha256(source) != sha256(destination):
            raise ValueError("retained calibration source differs; use a new experiment")
    atomic_json(OUTPUT/"editor_preflight.json",editor)
    sys.argv += ["--output-dir",str(OUTPUT),"--safety-predicate","gate_task"]
    placement = CpuPlacement()
    try:
        with placement:
            calibrate()
    finally:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
