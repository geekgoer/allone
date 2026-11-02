"""Run the frozen model-1 golden-vector test against the FLCN/v2 FPGA service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from falcon12_fpga_udp_client import Falcon12FpgaUdpClient


GOLDEN_X_NORMALIZED = np.asarray(
    [
        0.596082866,
        -0.076489605,
        -1.55807531,
        -0.235653177,
        0.521221817,
        0.278125644,
        -0.287052393,
        0.651442647,
        0.456826329,
        0.373339087,
        1.08079779,
        0.352927774,
    ],
    dtype=np.float32,
)
GOLDEN_INT8_PHYSICAL = np.asarray(
    [10.3390636, -0.909802377, 1.43537498, 0.0375078842],
    dtype=np.float64,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.77.2")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--timeout-ms", type=float, default=1000.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--model-id", type=int, default=1, choices=[1])
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dnn_runs/falcon12_fpga/falcon12_udp_smoke_model1.json"),
    )
    args = parser.parse_args()
    with Falcon12FpgaUdpClient(
        args.host,
        args.port,
        args.timeout_ms,
        args.retries,
        model_id=args.model_id,
    ) as client:
        result = client.infer_normalized(GOLDEN_X_NORMALIZED)
        if not result.fresh or result.outputs_physical is None:
            report = {
                "status": "FAIL",
                "protocol": "FLCN/v2",
                "server": {"host": args.host, "port": args.port},
                "model_id": args.model_id,
                "expected_int8_physical": GOLDEN_INT8_PHYSICAL.tolist(),
                "board_physical": None,
                "difference": None,
                "max_abs": None,
                "tolerance": args.tolerance,
                "transport": result.to_dict(),
                "client_stats": dict(client.stats),
            }
        else:
            difference = result.outputs_physical - GOLDEN_INT8_PHYSICAL
            max_abs = float(np.max(np.abs(difference)))
            no_retry = result.retries == 0
            report = {
                "status": "PASS" if max_abs <= args.tolerance and no_retry else "FAIL",
                "protocol": "FLCN/v2",
                "server": {"host": args.host, "port": args.port},
                "model_id": args.model_id,
                "expected_int8_physical": GOLDEN_INT8_PHYSICAL.tolist(),
                "board_physical": result.outputs_physical.tolist(),
                "difference": difference.tolist(),
                "max_abs": max_abs,
                "tolerance": args.tolerance,
                "no_retry_required": True,
                "transport": result.to_dict(),
                "client_stats": dict(client.stats),
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {args.report.resolve()}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
