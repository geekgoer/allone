"""Verify unattended UE4/AirSim startup without clicking Play."""

from __future__ import annotations

import argparse
import time

from airsim_process import AirSimLaunchConfig, AirSimProcessManager


DEFAULT_EDITOR = r"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"
DEFAULT_PROJECT = r"F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--editor", default=DEFAULT_EDITOR)
    parser.add_argument("--uproject", default=DEFAULT_PROJECT)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    if args.duration < 0.0:
        raise ValueError("duration must be non-negative")
    config = AirSimLaunchConfig.from_paths(
        args.editor,
        args.uproject,
        log_dir=args.log_dir,
        startup_timeout=args.startup_timeout,
    )
    manager = AirSimProcessManager(config)
    try:
        print(f"Launching UE4 game map: {config.map_asset}")
        print("Command: " + " ".join(manager.build_command()))
        client = manager.start_and_wait()
        client.confirmConnection()
        print(f"AirSim RPC ready without Play click (pid={manager.pid}, log={manager.log_path}).")
        if not args.keep_running:
            time.sleep(args.duration)
    finally:
        if not args.keep_running:
            manager.stop()
            print("Managed UE4/AirSim process stopped.")


if __name__ == "__main__":
    main()
