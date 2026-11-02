"""Explicit 300 -> 237 feature ablation; shared by offline and live inference."""

import numpy as np

from .decision_dataset import build_features


CONTRACT = "decision_convex_local_relative_pv_237d_v1"
REMOVED = tuple(range(3)) + tuple(220 + 4*i + j for i in range(20) for j in (1, 2, 3))
KEPT = tuple(i for i in range(300) if i not in REMOVED)


def project(feature):
    value = np.asarray(feature)
    if value.ndim not in (1, 2) or value.shape[-1] != 300:
        raise ValueError("projection requires a 300D feature or batch")
    return value[..., KEPT].copy()


def encode(state, references):
    return project(build_features(state, references))
