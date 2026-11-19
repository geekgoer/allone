"""Summarize native and delay-augmented end-to-end response times by DNN."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.dataset.resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("valid_for_svr", "").lower() == "true":
                groups[row["model_name"]].append(row)
    if not groups:
        raise ValueError("dataset has no valid_for_svr=true rows")

    summary = []
    for model_name, rows in sorted(groups.items()):
        r_max = np.asarray([float(row["R_max_ms"]) for row in rows], dtype=np.float64)
        r_p99 = np.asarray([float(row["R_p99_ms"]) for row in rows], dtype=np.float64)
        native_rows = [
            row for row in rows
            if float(row.get("actuation_delay_ms", "nan")) == 0.0
        ]
        if not native_rows:
            raise ValueError(
                f"model {model_name} has no zero-delay rows; "
                "native WCET requires at least one actuation_delay_ms=0 episode"
            )
        native_r_max = np.asarray(
            [float(row["R_max_ms"]) for row in native_rows], dtype=np.float64
        )
        native_r_p99 = np.asarray(
            [float(row["R_p99_ms"]) for row in native_rows], dtype=np.float64
        )
        native_wc_et = float(np.max(native_r_max))
        summary.append(
            {
                "model_name": model_name,
                "episodes": len(rows),
                "native_zero_delay_episodes": len(native_rows),
                "R_p99_ms_max": float(np.max(r_p99)),
                "R_max_ms_p50": float(np.percentile(r_max, 50)),
                "R_max_ms_p95": float(np.percentile(r_max, 95)),
                "R_max_ms_max": float(np.max(r_max)),
                "native_R_p99_ms_max": float(np.max(native_r_p99)),
                "native_R_max_ms_p95": float(np.percentile(native_r_max, 95)),
                "native_R_max_ms_max": native_wc_et,
                "recommended_native_wc_et_ms": native_wc_et,
                "candidate_rule": "R_candidate_ms = recommended_native_wc_et_ms + actuation_delay_ms",
                "recommended_rule": "estimate native WCET from zero-delay rows; add configured delay for each candidate",
            }
        )
    output = {
        "dataset": str(args.dataset.resolve()),
        "definition": "R is host state acquisition to AirSim command dispatch; native WCET uses actuation_delay_ms=0 rows",
        "models": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
