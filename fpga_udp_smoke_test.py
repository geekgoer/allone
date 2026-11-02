"""Send the frozen normalized golden vector to the FLCN/v1 FPGA server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fpga_udp_client import FalconFpgaUdpClient


GOLDEN_X_NORMALIZED = np.asarray([
    1.16163731, 2.53170323, -1.46144331, -1.98914778, 2.28400493,
    -1.60078406, -0.246512681, 1.52726865, 0.593094707, -0.804031849,
    0.659008265, 2.06079769, 1.09506488, 2.53146029, -1.45798016,
    0.81345588, 2.52501845, -1.43881369, 0.263079911, 2.50586629,
    -1.40395451, -0.516228914, 2.4409349, -1.31383252, -1.46385002,
], dtype=np.float32)
GOLDEN_INT8_PHYSICAL = np.asarray(
    [12.1535444, -0.546839237, 1.23984075, -1.04302478], dtype=np.float64
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.77.2")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-ms", type=float, default=1000.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--report", type=Path, default=Path("dnn_runs/iid_path_v1_depth10_width512/fpga_udp_smoke.json"))
    args = parser.parse_args()
    with FalconFpgaUdpClient(args.host, args.port, args.timeout_ms, args.retries) as client:
        result = client.infer_normalized(GOLDEN_X_NORMALIZED, "golden", 0)
        if not result.fresh or result.outputs_physical is None:
            report = {
                "status": "FAIL",
                "server": {"host": args.host, "port": args.port},
                "expected_int8_physical": GOLDEN_INT8_PHYSICAL.tolist(),
                "board_physical": None,
                "difference": None,
                "max_abs": None,
                "tolerance": args.tolerance,
                "transport": result.to_dict(),
                "client_stats": client.stats,
            }
        else:
            difference = result.outputs_physical - GOLDEN_INT8_PHYSICAL
            max_abs = float(np.max(np.abs(difference)))
            report = {
                "status": "PASS" if max_abs <= args.tolerance else "FAIL",
                "server": {"host": args.host, "port": args.port},
                "expected_int8_physical": GOLDEN_INT8_PHYSICAL.tolist(),
                "board_physical": result.outputs_physical.tolist(),
                "difference": difference.tolist(),
                "max_abs": max_abs,
                "tolerance": args.tolerance,
                "transport": result.to_dict(),
                "client_stats": client.stats,
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {args.report.resolve()}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
