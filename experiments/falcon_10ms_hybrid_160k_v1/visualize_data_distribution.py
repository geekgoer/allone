"""Create an honest distribution audit for the 10 m/s hybrid teacher data.

This script never changes teacher rows.  It can compare the completed 40k
AirSim core with any available extension dataset, including an incomplete
chunk directory.  A random visual subsample is used only for scatter plots;
all reported statistics and histograms use every available row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
LOWER = np.asarray([2.0, -3.0, -3.0, -2.0], dtype=np.float64)
UPPER = np.asarray([20.0, 3.0, 3.0, 2.0], dtype=np.float64)
BUCKET_NAMES = ("interior", "near_limit", "saturated")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")  # Okabe-Ito


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_chunks(root: Path) -> dict[str, np.ndarray] | None:
    chunks = sorted((root / "chunks").glob("chunk_*.npz"))
    if not chunks:
        return None
    merged: dict[str, list[np.ndarray]] = {}
    for path in chunks:
        for key, value in _load_npz(path).items():
            merged.setdefault(key, []).append(value)
    return {key: np.concatenate(values, axis=0) for key, values in merged.items()}


def _physical_bucket(u: np.ndarray) -> np.ndarray:
    at_limit = np.any((np.abs(u - LOWER) <= 1e-3) | (np.abs(u - UPPER) <= 1e-3), axis=1)
    span = UPPER - LOWER
    near = np.any((u - LOWER <= 0.15 * span) | (UPPER - u <= 0.15 * span), axis=1)
    return np.where(at_limit, 2, np.where(near, 1, 0)).astype(np.int8)


def _percentiles(value: np.ndarray) -> dict[str, float]:
    return {name: float(np.percentile(value, q)) for name, q in (("min", 0), ("p01", 1), ("p50", 50), ("p99", 99), ("max", 100))}


def _audit(name: str, data: dict[str, np.ndarray]) -> dict[str, Any]:
    x = np.asarray(data["X"], dtype=np.float64)
    u = np.asarray(data["U"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 12 or u.ndim != 2 or u.shape[1] != 4:
        raise ValueError(f"{name}: expected X=(N,12), U=(N,4), got {x.shape}/{u.shape}")
    bucket = np.asarray(data.get("control_bucket", _physical_bucket(u)), dtype=np.int8)
    if np.any(bucket < 0):
        bucket = _physical_bucket(u)
    return {
        "name": name,
        "samples": int(len(x)),
        "finite": bool(np.isfinite(x).all() and np.isfinite(u).all()),
        "reference_position_error_m": _percentiles(np.linalg.norm(x[:, 0:3], axis=1)),
        "reference_attitude_error_rad": _percentiles(np.linalg.norm(x[:, 3:6], axis=1)),
        "reference_velocity_error_mps": _percentiles(np.linalg.norm(x[:, 6:9], axis=1)),
        "reference_acceleration_mps2": _percentiles(np.linalg.norm(x[:, 9:12], axis=1)),
        "control_bucket_counts": {BUCKET_NAMES[i]: int(np.count_nonzero(bucket == i)) for i in range(3)},
        "control_bucket_fraction": {BUCKET_NAMES[i]: float(np.mean(bucket == i)) for i in range(3)},
        "control_per_output": {
            name: _percentiles(u[:, index])
            for index, name in enumerate(("thrust", "body_rate_x", "body_rate_y", "body_rate_z"))
        },
    }


def _plot(sets: list[tuple[str, dict[str, np.ndarray]]], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
        panels = (
            (axes[0, 0], slice(0, 3), "||p_ref - p|| (m)", (0, 12)),
            (axes[0, 1], slice(3, 6), "||rotation error|| (rad)", (0, 1.5)),
            (axes[0, 2], slice(6, 9), "||v_ref - v|| (m/s)", (0, 15)),
            (axes[1, 0], slice(9, 12), "||a_ref|| (m/s²)", (0, 15)),
        )
        for color, (name, data) in zip(COLORS, sets):
            x = np.asarray(data["X"], dtype=np.float64)
            for axis, dimensions, label, limits in panels:
                value = np.linalg.norm(x[:, dimensions], axis=1)
                axis.hist(value, bins=80, range=limits, density=True, histtype="step", linewidth=1.7, color=color, label=name)
                axis.set(xlabel=label, ylabel="density", xlim=limits)
        for axis, _, _, _ in panels:
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)

        bucket_axis = axes[1, 1]
        locations = np.arange(3)
        width = 0.78 / max(1, len(sets))
        for index, (name, data) in enumerate(sets):
            u = np.asarray(data["U"], dtype=np.float64)
            bucket = np.asarray(data.get("control_bucket", _physical_bucket(u)), dtype=np.int8)
            if np.any(bucket < 0):
                bucket = _physical_bucket(u)
            counts = np.asarray([np.mean(bucket == value) for value in range(3)])
            bucket_axis.bar(locations - 0.39 + width / 2 + index * width, counts, width=width, color=COLORS[index], label=name)
        bucket_axis.set(xticks=locations, xticklabels=BUCKET_NAMES, ylabel="fraction of rows", ylim=(0, 1), title="Control difficulty bucket")
        bucket_axis.legend(fontsize=8)
        bucket_axis.grid(axis="y", alpha=0.2)

        scatter_axis = axes[1, 2]
        rng = np.random.default_rng(20260917)
        for color, (name, data) in zip(COLORS, sets):
            x = np.asarray(data["X"], dtype=np.float64)
            count = min(12000, len(x))
            selected = rng.choice(len(x), size=count, replace=False)
            scatter_axis.scatter(x[selected, 1], x[selected, 2], s=2, alpha=0.16, color=color, label=name, rasterized=True)
        scatter_axis.axhline(0, color="#555555", linewidth=0.6)
        scatter_axis.axvline(0, color="#555555", linewidth=0.6)
        scatter_axis.set(xlabel="lateral position error (m)", ylabel="vertical position error (m)", xlim=(-5, 5), ylim=(-5, 5), title="Reference error visual subsample")
        scatter_axis.legend(fontsize=8, markerscale=3)
        scatter_axis.grid(alpha=0.2)
        fig.suptitle("10 m/s hybrid teacher data distribution audit\nHistograms use all rows; scatter uses a fixed-seed visual subsample.", fontsize=14)
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=HERE / "core_40k_airsim_acado_10ms_v2_safe_start" / "final" / "falcon_hybrid_core_40k_10ms_v2_safe_start.npz")
    parser.add_argument("--failed-v2-root", type=Path, default=HERE / "extensions_120k_phase_consistent_10ms_v2_safe_start")
    parser.add_argument("--v3-smoke", type=Path, default=HERE / "extensions_120k_phase_consistent_10ms_v3_bucket_balanced_smoke" / "final" / "falcon_hybrid_extensions_smoke.npz")
    parser.add_argument("--v3-final", type=Path, default=HERE / "extensions_120k_phase_consistent_10ms_v4_environment_balanced" / "final" / "falcon_hybrid_extensions_120k_10ms_v4_environment_balanced.npz")
    parser.add_argument("--output-dir", type=Path, default=HERE / "diagnostics")
    args = parser.parse_args()

    sets: list[tuple[str, dict[str, np.ndarray]]] = [("40k AirSim core", _load_npz(args.core.resolve()))]
    failed = _load_chunks(args.failed_v2_root.resolve())
    if failed is not None:
        sets.append(("old v2 partial (failed)", failed))
    if args.v3_smoke.exists():
        sets.append(("new v3 smoke", _load_npz(args.v3_smoke.resolve())))
    if args.v3_final.exists():
        sets.append(("new v3 full", _load_npz(args.v3_final.resolve())))

    output_dir = args.output_dir.resolve()
    report = {"provenance": {"core": str(args.core.resolve()), "v2_partial_root": str(args.failed_v2_root.resolve()), "v3_smoke": str(args.v3_smoke.resolve()), "v3_final": str(args.v3_final.resolve())}, "datasets": [_audit(name, data) for name, data in sets]}
    (output_dir / "data_distribution_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_distribution_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot(sets, output_dir / "data_distribution.png")
    print(json.dumps(report, indent=2))
    print(f"Figure: {output_dir / 'data_distribution.png'}")
    print(f"Summary: {output_dir / 'data_distribution_summary.json'}")


if __name__ == "__main__":
    main()
