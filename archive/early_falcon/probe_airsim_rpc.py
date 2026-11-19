"""Probe the AirSim RPC operations used by teacher-data collection.

This is a diagnostic utility, not part of the dataset itself.  It launches the
same unattended UE4 command as the collector and reports the first operation
that fails, including its original exception and elapsed time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable

from airsim_interface import require_airsim
from airsim_process import AirSimLaunchConfig, AirSimProcessManager
from collect_teacher_dataset import load_collection_config, _resolve_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher_collection.json"


def _call(label: str, function: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    try:
        result = function()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"FAIL {label}: {type(exc).__name__}: {exc} ({elapsed:.2f}s)")
        raise
    elapsed = time.perf_counter() - started
    print(f"OK   {label} ({elapsed:.2f}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--list-assets-only", action="store_true")
    args = parser.parse_args()
    config, config_path = load_collection_config(args.config)
    base_dir = config_path.parent.parent
    airsim_config = config["airsim"]
    launch_config = AirSimLaunchConfig.from_paths(
        airsim_config["editor"],
        airsim_config["uproject"],
        map_asset=airsim_config.get("map_asset"),
        log_dir=_resolve_path(config.get("log_dir", "logs"), base_dir),
        startup_timeout=float(airsim_config["startup_timeout"]),
        retry_interval=float(airsim_config["retry_interval"]),
        rpc_host=str(airsim_config.get("rpc_host", "127.0.0.1")),
        rpc_port=int(airsim_config.get("rpc_port", 41451)),
        rpc_timeout=float(airsim_config.get("rpc_timeout", 30.0)),
        res_x=int(airsim_config["res_x"]),
        res_y=int(airsim_config["res_y"]),
    )
    manager = AirSimProcessManager(launch_config)
    client = None
    try:
        client = manager.start_and_wait()
        print(f"RPC ready: pid={manager.pid}, log={manager.log_path}")
        airsim = require_airsim()
        vehicle_name = str(config["vehicle_name"])
        pose = _call(
            "simGetVehiclePose",
            lambda: client.simGetVehiclePose(vehicle_name=vehicle_name),
        )
        assets = _call("simListAssets", client.simListAssets)
        assets = [str(asset) for asset in assets]
        print(
            f"Asset registry: count={len(assets)}, "
            f"contains_{config['gate']['asset']}={str(config['gate']['asset']) in assets}"
        )
        if args.list_assets_only:
            for asset in assets:
                print(f"  {asset}")
            return
        _call("reset", client.reset)
        _call("simPause(True)", lambda: client.simPause(True))
        _call(
            "simSetVehiclePose",
            lambda: client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name),
        )
        probe_pose = airsim.Pose(
            position_val=airsim.Vector3r(0.0, 0.0, -2.0),
            orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
        )
        probe_name = "FalconRPCProbe"
        _call(
            "simSpawnObject",
            lambda: client.simSpawnObject(
                probe_name,
                str(config["gate"]["asset"]),
                probe_pose,
                airsim.Vector3r(1.0, 1.0, 1.0),
                False,
                False,
            ),
        )
        _call("simDestroyObject", lambda: client.simDestroyObject(probe_name))
        _call("simPause(False)", lambda: client.simPause(False))
        _call(
            "simGetCollisionInfo",
            lambda: client.simGetCollisionInfo(vehicle_name=vehicle_name),
        )
        print("AirSim RPC probe passed.")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
