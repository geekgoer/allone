"""Low-speed simulated contact check for the same mesh used by gate frames."""

from datetime import datetime, timezone
import time

from airsim_interface import require_airsim, vehicle_local_to_world_offset
from .decision_dataset import ROOT, atomic_json
from .fly_convex_policy import load_airsim_client
from .framework_runtime_editor import LOG
from .framework_runtime_world import FrameworkWorld


def main():
    load_airsim_client()
    api = require_airsim()
    client = api.MultirotorClient(timeout_value=10.)
    directory = ROOT/"controller_architecture_study/results/gate_contact_audit"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True,exist_ok=False)
    wall,trace,error = None,[],None
    report = {"purpose":"collision plumbing check, NOT DNN control or scheduling comparison",
        "asset":"1M_Cube","speed_mps":1.,"FPGA_verified":False}
    try:
        if client.simListSceneObjects("FalconSafetyGate_.*"):
            raise RuntimeError("another gate experiment is active; do not disturb it")
        if client.simIsPause():
            raise RuntimeError("Play must be running and unpaused")
        client.reset()
        client.enableApiControl(True,vehicle_name="Drone1")
        client.armDisarm(True,vehicle_name="Drone1")
        client.takeoffAsync(vehicle_name="Drone1").join()
        client.moveToPositionAsync(0.,0.,-3.,1.,vehicle_name="Drone1").join()
        client.hoverAsync(vehicle_name="Drone1").join()
        offset = vehicle_local_to_world_offset(client,"Drone1")
        state = client.getMultirotorState(vehicle_name="Drone1")
        p = state.kinematics_estimated.position
        world = [p.x_val+offset[0]+3.,p.y_val+offset[1],p.z_val+offset[2]]
        wall = client.simSpawnObject("FalconCollisionAudit_wall","1M_Cube",
            api.Pose(api.Vector3r(*map(float,world)),api.Quaternionr(0.,0.,0.,1.)),
            api.Vector3r(.15,2.,2.),False,False)
        report.update(object_name=wall,initial_local_position=[p.x_val,p.y_val,p.z_val],wall_world_position=world,
            world_offset=offset.tolist(),initial_collision_stamp=int(state.collision.time_stamp))
        offset_in_log = LOG.stat().st_size
        queries = [f"getall StaticMeshComponent {prop} OUTER={wall}"
            for prop in ("BodyInstance","StaticMesh","RelativeScale3D")]
        queries.append("getall StaticMesh BodySetup NAME=1M_Cube")
        for query in queries:
            client.simRunConsoleCommand(query)
        time.sleep(.05)
        with LOG.open("rb") as handle:
            handle.seek(offset_in_log)
            atomic_json(directory/"collision_properties.json",{"queries":queries,
                "log":handle.read().decode("utf-8",errors="replace")})
        client.moveByVelocityAsync(1.,0.,0.,5.,vehicle_name="Drone1")
        began = time.perf_counter()
        while time.perf_counter()-began < 5.5:
            s = client.getMultirotorState(vehicle_name="Drone1")
            c = client.simGetCollisionInfo(vehicle_name="Drone1")
            position = s.kinematics_estimated.position
            row = {"wall_s":time.perf_counter()-began,"position":[position.x_val,position.y_val,position.z_val],
                "state_collision":bool(s.collision.has_collided),"state_object":s.collision.object_name,
                "state_stamp":int(s.collision.time_stamp),"direct_collision":bool(c.has_collided),
                "direct_object":c.object_name,"direct_stamp":int(c.time_stamp)}
            trace.append(row)
            if ((row["state_collision"] and row["state_object"]==wall)
                    or (row["direct_collision"] and row["direct_object"]==wall)):
                break
            time.sleep(.02)
        state_hit = any(r["state_collision"] and r["state_object"]==wall for r in trace)
        direct_hit = any(r["direct_collision"] and r["direct_object"]==wall for r in trace)
        _,verified,_ = FrameworkWorld.snapshot_from(client,"Drone1")
        snapshot_hit = bool(verified.object_name==wall and int(verified.time_stamp)>report["initial_collision_stamp"])
        report.update(state_packet_contact_observed=state_hit,direct_api_contact_observed=direct_hit,
            corrected_snapshot_contact_observed=snapshot_hit,
            passed=direct_hit and snapshot_hit,final_sample=trace[-1] if trace else None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.hoverAsync(vehicle_name="Drone1")
            client.ping()
            report["object_removed"] = bool(wall and client.simDestroyObject(wall))
        except Exception as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        client.client.close()
        report["error"] = error
        atomic_json(directory/"trace.json",trace)
        atomic_json(directory/"report.json",report)
    print(f"CONTACT_AUDIT {report} Report: {directory}",flush=True)


if __name__ == "__main__":
    main()
