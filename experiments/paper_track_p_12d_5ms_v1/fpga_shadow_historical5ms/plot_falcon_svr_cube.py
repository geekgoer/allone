"""Plot paper-style FALCON SVR response cubes for fixed environment difficulty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def _predict(payload: dict, x: np.ndarray) -> np.ndarray:
    scale = np.asarray(payload["feature_maximum_for_c_T_R"], dtype=np.float64)
    transformed = x.astype(np.float64, copy=True)
    transformed[:, :3] /= scale
    return np.asarray(payload["model"].predict(transformed), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.8)
    args = parser.parse_args()
    payload = joblib.load(args.model.resolve())
    with args.dataset.resolve().open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("valid_for_svr", "").lower() == "true"]
    rows = [row for row in rows if np.isclose(float(row["epsilon"]), args.epsilon)]
    if not rows:
        raise ValueError(f"dataset contains no valid rows for epsilon={args.epsilon}")
    c = np.asarray([1.0 / float(row["beta_test_normalized_mse"]) for row in rows])
    t = np.asarray([float(row["period_ms"]) for row in rows])
    r = np.asarray([float(row["R_max_ms"]) for row in rows])
    j = np.asarray([float(row["J_paper_style"]) for row in rows])
    predicted = _predict(payload, np.column_stack((c, t, r, np.full(c.size, args.epsilon))))
    scale = np.asarray(payload["feature_maximum_for_c_T_R"], dtype=np.float64)
    cn, tn, rn = c / scale[0], t / scale[1], r / scale[2]
    color_norm = Normalize(vmin=float(min(j.min(), predicted.min())), vmax=float(max(j.max(), predicted.max())))
    fig = plt.figure(figsize=(13, 6), dpi=180, layout="constrained")
    axes = []
    for index, (values, title) in enumerate(((j, "Measured J"), (predicted, "SVR predicted J")), start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        scatter = ax.scatter(cn, tn, rn, c=values, cmap="turbo", norm=color_norm, s=22, depthshade=False)
        ax.set_xlabel("c / c_max")
        ax.set_ylabel("T / T_max")
        ax.set_zlabel("R / R_max")
        upper = max(1.0, float(max(cn.max(), tn.max(), rn.max())))
        ax.set(xlim=(0, upper), ylim=(0, upper), zlim=(0, upper))
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(title)
        ax.view_init(elev=25, azim=-55)
        axes.append(ax)
    fig.colorbar(scatter, ax=axes, shrink=0.65, pad=0.08, label="Performance J (shared scale)")
    fig.suptitle(f"FALCON Fig. 5-style cubes, epsilon={args.epsilon:g}\nPredictions at measured coordinates; unsampled regions are not filled")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"falcon_performance_cube_epsilon_{str(args.epsilon).replace('.', 'p')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    report = {"model": str(args.model.resolve()), "dataset": str(args.dataset.resolve()), "epsilon": args.epsilon, "figure": str(path.resolve()), "axes": {"x": "c normalized", "y": "T normalized", "z": "R normalized", "color": "J"}, "observed_J_range": [float(j.min()), float(j.max())], "predicted_J_range": [float(predicted.min()), float(predicted.max())], "points": int(len(rows))}
    (args.output_dir / "falcon_performance_cube_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
