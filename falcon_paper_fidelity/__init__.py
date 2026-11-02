"""Paper-fidelity FALCON reproduction components.

This package is intentionally isolated from the earlier engineering baseline.
The public paper protocol and the robustness extensions have different output
roots and manifests so their samples and metrics cannot be mixed silently.
"""

from .contracts import INPUT_CONTRACT, INPUT_FEATURES, OUTPUT_FEATURES, build_gate_input
from .motion_primitive import MotionPrimitiveConfig, QuinticMotionPrimitivePlanner

__all__ = (
    "INPUT_CONTRACT",
    "INPUT_FEATURES",
    "OUTPUT_FEATURES",
    "MotionPrimitiveConfig",
    "QuinticMotionPrimitivePlanner",
    "build_gate_input",
)
