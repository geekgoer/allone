"""Identical sensor service for both policies; disturbance records never enter its data pipe."""

import gc
import math
import multiprocessing
import time

import numpy as np

from airsim_interface import require_airsim
from .framework_cpu_affinity import set_process_cpus
from .framework_runtime_io import CommandIO
from .framework_runtime_io_wire import IOConnection
from .framework_runtime_wait import DeadlineWaiter, CompletionEvent


SENSOR_PERIOD = .020


def projected_margin(risk, radius=None):
    if radius is None:
        return risk["h"]+.5*min(0.,risk["hdot"])
    return min(risk["h"],radius-abs(radius-risk["h"]-.5*risk["hdot"]))


def boundary_urgency(risk, radius):
    # Inward motion can reach the opposite wall after crossing the center.
    distance=risk["h"] if risk["hdot"]<=0. else 2*radius-risk["h"]
    return .5*abs(risk["hdot"])/max(.025,distance)


def event_edge(armed, risk, *, radius=None):
    margin = projected_margin(risk,radius)
    if armed and margin<=0.:
        return True,False
    return False,armed or margin>.05


def inject_velocity(client, pulse, initial_ns, origin):
    """A diagnostic velocity jump, not a wind model or a new pose target."""
    began=time.perf_counter()
    before=client.simGetGroundTruthKinematics(vehicle_name=pulse["vehicle"])
    p=np.array([before.position.x_val,before.position.y_val,before.position.z_val])
    velocity=np.array([before.linear_velocity.x_val,before.linear_velocity.y_val,before.linear_velocity.z_val])
    before.linear_velocity.y_val += pulse["delta_y_mps"]
    read_done=time.perf_counter()
    client.simSetKinematics(before,ignore_collision=False,vehicle_name=pulse["vehicle"])
    returned=time.perf_counter()
    after=client.simGetGroundTruthKinematics(vehicle_name=pulse["vehicle"])
    stamp=client.getMultirotorState(vehicle_name=pulse["vehicle"]).timestamp
    actual_p=np.array([after.position.x_val,after.position.y_val,after.position.z_val])
    return {"pulse":pulse,"started_s":began,"snapshot_received_s":read_done,"rpc_return_s":returned,
        "wall_elapsed_s":returned-origin,"post_simulation_s":(int(stamp)-initial_ns)/1e9,
        "position_before":p.tolist(),"position_after":actual_p.tolist(),
        "velocity_before":velocity.tolist(),"requested_velocity":(velocity+[0.,pulse["delta_y_mps"],0.]).tolist(),
        "position_displacement_m":float(np.linalg.norm(actual_p-p)),
        "estimated_pose_writeback_bound_m":float(np.linalg.norm(velocity)*(returned-began)),
        "ignore_collision":False,"pose_target_changed":False,"physics_paused":False}


def sensor_worker(control, data, vehicles, calibration, cpus):
    data=IOConnection(data)
    io=None
    injector=None
    completion=None
    try:
        placement=set_process_cpus(cpus)
        from .fly_convex_policy import load_airsim_client
        load_airsim_client()
        completion=CompletionEvent()
        io=CommandIO(len(vehicles),calibration,.003,cpus,completed=completion.set)
        injector=require_airsim().MultirotorClient(timeout_value=3.)
        injector.ping()
        gc.collect()
        gc.disable()
        control.send(("ready",placement))
        kind,origin,duration,initial_ns,pulse=control.recv()
        if kind!="start":
            raise ValueError("sensor service expected a start timestamp")
        slots={v:0 for v in vehicles}
        covered={v:False for v in vehicles}
        phases={v:i*SENSOR_PERIOD/len(vehicles) for i,v in enumerate(vehicles)}
        injections,skipped=[],0
        next_pulse_check=origin+float(pulse["time_s"])-.030 if pulse else float("inf")
        stopping=False
        with DeadlineWaiter() as timer:
            while not stopping:
                if control.poll():
                    if control.recv()[0]!="stop":
                        raise ValueError("unexpected sensor command")
                    break
                now=time.perf_counter()
                if pulse and not injections and now>=next_pulse_check:
                    packet=injector.getMultirotorState(vehicle_name=pulse["vehicle"])
                    elapsed=(int(packet.timestamp)-initial_ns[pulse["vehicle"]])/1e9
                    if elapsed>=pulse["time_s"]:
                        injections.append(inject_velocity(injector,pulse,initial_ns[pulse["vehicle"]],origin))
                        next_pulse_check=float("inf")
                    else:
                        next_pulse_check=time.perf_counter()+min(.003,max(.0005,pulse["time_s"]-elapsed))
                for packet,row in io.drain_samples():
                    covered[row["vehicle"]]=(packet[0].timestamp_ns-initial_ns[row["vehicle"]])/1e9>=duration
                    row["status"]="monitored"
                    row["sample_message_send_s"]=time.perf_counter()
                    data.send(("sample_done",packet,row))
                now=time.perf_counter()
                for v in vehicles:
                    due=origin+phases[v]+slots[v]*SENSOR_PERIOD
                    if v not in io.samples and due<=now and not covered[v]:
                        index=slots[v]
                        row={"vehicle":v,"sample_id":f"sensor:{v}:{index}","model":"sensor_only",
                            "deadline_s":due+SENSOR_PERIOD,"status":"sampling"}
                        io.sample(row)
                        following=max(index+1,math.ceil((now-origin-phases[v])/SENSOR_PERIOD))
                        skipped+=following-index-1
                        slots[v]=following
                wake=min([time.perf_counter()+.100,next_pulse_check,
                    *[origin+phases[v]+slots[v]*SENSOR_PERIOD for v in vehicles
                        if v not in io.samples and not covered[v]]])
                timer.wait([control,completion.handle],wake)
        io.executor.shutdown(wait=True,cancel_futures=False)
        for packet,row in io.drain_samples():
            row["status"]="monitored"
            row["sample_message_send_s"]=time.perf_counter()
            data.send(("sample_done",packet,row))
        io.close()
        io=None
        control.send(("closed",{"placement":placement,"injections":injections,
            "sensor_skipped_releases":skipped,"transport_stalls":data.stalls}))
    except BaseException as exc:
        control.send(("error",f"{type(exc).__name__}: {exc}"))
    finally:
        if io is not None:
            io.close()
        if injector is not None:
            injector.client.close()
        if completion is not None:
            completion.close()
        control.close()
        data.close()


class InflightSensing:
    def __init__(self, vehicles, calibration, cpu, pulse):
        self.vehicles,self.calibration,self.cpu,self._pulse=vehicles,calibration,cpu,pulse
        self.process,self.connection,self.control=None,None,None
        self.records,self.summary,self.latest_ns=[],None,{}
        self.placement={"initialization_complete":False}

    def prepare(self):
        context=multiprocessing.get_context("spawn")
        self.control,child_control=context.Pipe()
        receiver,sender=context.Pipe(duplex=False)
        self.connection=IOConnection(receiver)
        self.process=context.Process(target=sensor_worker,
            args=(child_control,sender,self.vehicles,self.calibration,[self.cpu]),daemon=True)
        self.process.start()
        child_control.close()
        sender.close()
        if not self.control.poll(30.):
            raise RuntimeError("sensor initialization timed out")
        message=self.control.recv()
        if message[0]!="ready":
            raise RuntimeError(str(message))
        self.placement=message[1]

    def start(self, origin, duration, initial):
        self.control.send(("start",origin,duration,{v:initial[v]["sim_ns"] for v in self.vehicles},self._pulse))

    def drain(self, maximum=7):
        result=[]
        for _ in range(maximum):
            if not self.connection.poll():
                break
            kind,packet,row=self.connection.recv()
            if kind!="sample_done":
                raise RuntimeError("sensor data channel contained non-state information")
            record={"packet":packet,"row":row,"received_s":time.perf_counter()}
            self.records.append(record)
            self.latest_ns[row["vehicle"]]=max(self.latest_ns.get(row["vehicle"],0),packet[0].timestamp_ns)
            result.append(record)
        if self.control.poll():
            message=self.control.recv()
            raise RuntimeError(f"sensor stopped during flight: {message}")
        return result

    def covers(self, initial, duration):
        return all((self.latest_ns.get(v,0)-initial[v]["sim_ns"])/1e9>=duration for v in self.vehicles)

    def close(self):
        if self.process is None:
            return
        self.control.send(("stop",))
        until=time.perf_counter()+10.
        data_closed=False
        while time.perf_counter()<until:
            while not data_closed and self.connection.poll():
                try:
                    _,packet,row=self.connection.recv()
                except EOFError:
                    data_closed=True
                    break
                self.records.append({"packet":packet,"row":row,"received_s":time.perf_counter()})
                self.latest_ns[row["vehicle"]]=max(self.latest_ns.get(row["vehicle"],0),packet[0].timestamp_ns)
            if self.control.poll(.001):
                message=self.control.recv()
                if message[0]!="closed":
                    raise RuntimeError(str(message))
                self.summary=message[1]
                break
        self.process.join(2.)
        if self.process.is_alive():
            self.process.terminate()  # only our private sensor helper
            self.process.join(2.)
            raise RuntimeError("sensor helper failed to close")
        self.connection.close()
        self.control.close()
        if self.summary is None:
            raise RuntimeError("sensor helper provided no final record")
