"""Read-only state-RPC diagnostic; never sends flight commands or resets."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import socket
import sys
import threading
import time

import numpy as np

from airsim_interface import require_airsim
from .decision_dataset import ROOT, atomic_json
from .fly_convex_policy import load_airsim_client


def main():
    load_airsim_client()
    api = require_airsim()
    sys.setswitchinterval(.0005)
    gc.collect()
    gc.disable()
    barrier = threading.Barrier(3)
    def probe(index):
        client = api.MultirotorClient(timeout_value=3.)
        rows = []
        try:
            client.ping()
            stream = client.client._transport._sockets[0]._stream
            for block,flag in enumerate((0,1,1,0)):
                stream.socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,flag)
                barrier.wait(10.)
                origin = time.perf_counter()
                for k in range(40):
                    time.sleep(max(0.,origin+k*.030-time.perf_counter()))
                    began = time.perf_counter()
                    state = client.getMultirotorState(vehicle_name=f"Drone{index+1}")
                    rows.append({"block":block,"nodelay":flag,"vehicle":index+1,
                        "rtt_ms":1000*(time.perf_counter()-began),"sim_ns":int(state.timestamp)})
            return rows
        finally:
            client.client.close()
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = [row for group in executor.map(probe,range(3)) for row in group]
    summary = {}
    for flag in (0,1):
        values = [r["rtt_ms"] for r in rows if r["nodelay"] == flag]
        summary[str(flag)] = {"n":len(values),"mean_ms":float(np.mean(values)),
            "p95_ms":float(np.percentile(values,95)),"p99_ms":float(np.percentile(values,99)),"max_ms":max(values)}
    root = ROOT/"controller_architecture_study/results/framework_transport_probe_v2"
    target = root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+".json")
    atomic_json(target,{"rows":rows,"summary":summary,"scope":"three concurrent read-only state RPCs; idle Play, not a flight comparison",
        "controls_sent":0,"reset_called":False,"system_or_package_modified":False})
    print(summary,flush=True)
    print(f"Report: {target}",flush=True)


if __name__ == "__main__":
    main()
