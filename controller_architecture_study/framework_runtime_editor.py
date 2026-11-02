"""Read the live editor's background-throttle flag; never change preferences."""

from pathlib import Path
import re
import time


LOG = Path("F:/Airsim/AirSim/Unreal/Environments/Blocks/Saved/Logs/Blocks.log")
QUERY = "get EditorPerformanceSettings bThrottleCPUWhenNotForeground"


def parse_flag(text):
    values = re.findall(r"\]\s*(True|False)\s*$",text,re.M)
    return values[-1] == "True" if values else None


def read_background_throttle():
    from .fly_convex_policy import load_airsim_client
    from airsim_interface import require_airsim
    load_airsim_client()
    client = require_airsim().MultirotorClient(timeout_value=3.)
    try:
        offsets = {path:path.stat().st_size for path in LOG.parent.glob("Blocks*.log")}
        if not client.simRunConsoleCommand(QUERY):
            raise RuntimeError("editor setting query was rejected")
        until = time.perf_counter()+2.
        while time.perf_counter()<until:
            for path in LOG.parent.glob("Blocks*.log"):
                with path.open("rb") as handle:
                    handle.seek(offsets.get(path,0))
                    value = parse_flag(handle.read().decode("utf-8",errors="replace"))
                if value is not None:
                    return {"background_throttle_enabled":value,"query":QUERY,"log":str(path),
                        "preferences_modified":False,"all_windows_hidden_checked":False}
            time.sleep(.02)
        raise RuntimeError("editor query was not logged within2s; do not assume it is disabled")
    finally:
        client.client.close()
