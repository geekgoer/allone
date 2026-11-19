"""Re-run the exact previously successful frozen source, without new ablation code."""

import argparse
import sys

import controller_architecture_study
from .decision_dataset import ROOT, atomic_json


FROZEN = ROOT/"controller_architecture_study/results/feedback_residency_20261027_v2/full_sequence/frozen_source"
controller_architecture_study.__path__.insert(0, str(FROZEN))

from .framework_cpu_affinity import CpuPlacement
from . import probe_feedback_handoff as frozen_probe
from . import probe_seven_continuous_timing as frozen_flight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output)
    if Path(frozen_probe.__file__).parent != FROZEN or Path(frozen_flight.__file__).parent != FROZEN:
        raise RuntimeError("frozen modules were not actually loaded")
    placement = CpuPlacement()
    try:
        with placement:
            frozen_probe.run(output, placement, False, "category", resident_worker=True)
    finally:
        if output.exists():
            atomic_json(output/"cpu_placement.json", placement.report)
            atomic_json(output/"frozen_control.json", dict(source=str(FROZEN),
                probe_file=frozen_probe.__file__, flight_file=frozen_flight.__file__))


if __name__=="__main__":
    main()
