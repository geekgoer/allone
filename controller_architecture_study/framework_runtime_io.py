"""Bounded per-loop command I/O; never share a Tornado client across threads."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
import gc
import multiprocessing
import queue
import threading
import time
from multiprocessing.connection import wait

import numpy as np

from airsim_interface import AirSimDrone, require_airsim
from control_adapter import control_to_airsim_command, send_control, throttle_to_thrust
from .framework_runtime_world import FrameworkWorld
from .framework_cpu_affinity import set_thread_cpus, set_process_cpus
from .framework_runtime_io_wire import IOConnection
from .framework_runtime_wait import CompletionEvent


class CommandIO:
    def __init__(self, count, calibration, command_bound, cpus=None, notify=None, *, allow_late=False, completed=None):
        self.calibration,self.command_bound = calibration,command_bound
        self.allow_late = allow_late
        self.cpus,self.placement = cpus,[]
        self.notify = notify or (lambda row:None)
        self.completed = completed or (lambda:None)
        self.local,self.clients,self.futures,self.samples = threading.local(),[],{},{}
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=count,initializer=self.connect,thread_name_prefix="falcon-command-io")
        barrier = threading.Barrier(count)
        tasks = [self.executor.submit(barrier.wait,10.) for _ in range(count)]
        try:
            for task in tasks:
                task.result(timeout=15.)
        except BaseException:
            self.close()
            raise

    def connect(self):
        placement = set_thread_cpus(self.cpus) if self.cpus else {"enabled":False}
        client = require_airsim().MultirotorClient(timeout_value=3.)
        if not client.ping():
            raise RuntimeError("command I/O connection failed")
        client.getMultirotorState(vehicle_name="Drone1")  # warm full packet decoding, not only ping
        self.local.client = client
        with self.lock:
            self.clients.append(client)
            self.placement.append(placement)

    def send(self, action, row):
        v = row["vehicle"]
        if v in self.futures or v in self.samples:
            raise ValueError("two commands for one vehicle cannot be in flight")
        row.setdefault("io_submitted_s",time.perf_counter())
        self.futures[v] = self.executor.submit(self._send,action,row)
        self.futures[v].add_done_callback(lambda unused:self.completed())

    def sample(self, row):
        v = row["vehicle"]
        if v in self.futures or v in self.samples:
            raise ValueError("one I/O operation per control loop")
        row.setdefault("sample_submitted_s",time.perf_counter())
        self.samples[v] = self.executor.submit(self._sample,row)
        self.samples[v].add_done_callback(lambda unused:self.completed())

    def _sample(self, row):
        row["sample_io_started_s"] = time.perf_counter()
        self.notify(row)
        packet = FrameworkWorld.snapshot_from(self.local.client,row["vehicle"])
        row["sample_io_finished_s"] = time.perf_counter()
        return packet,row

    def drain_samples(self):
        ready = []
        for v,future in list(self.samples.items()):
            if future.done():
                packet,row=future.result()
                row["sample_helper_collected_s"]=time.perf_counter()
                ready.append((packet,row))
                del self.samples[v]
        return ready

    def _send(self, action, row):
        now = time.perf_counter()
        row["io_started_s"] = now
        self.notify(row)
        if self.stopping.is_set():
            row["status"] = "cancelled_before_command"
            return None,row
        if action is None or (not self.allow_late and now >= row["deadline_s"]):
            row["status"] = "expired_not_applied"
            return None,row
        if not self.allow_late and now+self.command_bound > row["deadline_s"]:
            row["status"] = "delivery_budget_exhausted"
            return None,row
        action = np.clip(action,[2.,-3.,-3.,-2.],[20.,3.,3.,2.])
        command = control_to_airsim_command(action,self.calibration,duration=1.)
        mapped = [throttle_to_thrust(command.throttle,self.calibration),command.roll_rate,-command.pitch_rate,-command.yaw_rate]
        row.update(command_send_s=now,control=action.tolist(),airsim_command=vars(command),mapped_control=mapped)
        client = self.local.client
        send_control(AirSimDrone(client,row["vehicle"]),action,self.calibration,duration=1.)
        client.ping()
        row.update(rpc_return_s=time.perf_counter(),status="applied")
        self.notify(row)
        packet = FrameworkWorld.snapshot_from(client,row["vehicle"])
        row["io_finished_s"] = time.perf_counter()
        return packet,row

    def drain(self):
        ready = []
        for v,future in list(self.futures.items()):
            if future.done():
                packet,row=future.result()
                row["command_helper_collected_s"]=time.perf_counter()
                ready.append((packet,row))
                del self.futures[v]
        return ready

    def close(self):
        self.stopping.set()
        self.executor.shutdown(wait=True,cancel_futures=True)
        for client in self.clients:
            client.client.close()


def io_worker(connection, count, calibration, command_bound, cpus, allow_late=False):
    connection = IOConnection(connection)
    io = None
    wake = None
    deliveries = []
    try:
        placement = set_process_cpus(cpus) if cpus else {"enabled":False}
        from .fly_convex_policy import load_airsim_client
        load_airsim_client()  # spawn starts a fresh interpreter, unlike parent threads
        events = queue.Queue()
        wake=CompletionEvent()
        def notify(row):
            events.put(dict(row))
            wake.set()
        io = CommandIO(count,calibration,command_bound,cpus,notify=notify,allow_late=allow_late,completed=wake.set)
        def deliver(kind,packet,row):
            began=time.perf_counter()
            row["sample_message_send_s" if kind=="sample_done" else "command_message_send_s"]=began
            connection.send((kind,packet,row))
            deliveries.append({"sample_id":row["sample_id"],"kind":kind,"start_s":began,"end_s":time.perf_counter()})
        gc.collect()
        gc.disable()
        connection.send(("ready",{"process":placement,"threads":io.placement}))
        stopping = False
        while not stopping:
            while connection.poll():
                message = connection.recv()
                if message[0] == "stop":
                    stopping = True
                    break
                if message[0] == "sample":
                    message[1]["sample_helper_received_s"] = time.perf_counter()
                    io.sample(message[1])
                elif message[0] == "send":
                    message[2]["command_helper_received_s"] = time.perf_counter()
                    io.send(message[1],message[2])
                else:
                    raise ValueError("unknown command helper request")
            samples,commands = io.drain_samples(),io.drain()
            # Completion futures synchronize with the producer. Flush their
            # earlier start/ack events BEFORE sending completion and retiring IDs.
            while not events.empty():
                connection.send(("update",events.get_nowait()))
            for packet,row in samples:
                deliver("sample_done",packet,row)
            for packet,row in commands:
                deliver("command_done",packet,row)
            if not stopping:
                wait([connection.connection,wake.handle])
        # Finish/record already in-flight RPCs; queued commands see stopping
        # before sending. Do not lose a late actuation from the final report.
        io.stopping.set()
        io.executor.shutdown(wait=True,cancel_futures=False)
        samples,commands = io.drain_samples(),io.drain()
        while not events.empty():
            connection.send(("update",events.get_nowait()))
        for packet,row in samples:
            deliver("sample_done",packet,row)
        for packet,row in commands:
            deliver("command_done",packet,row)
        io.close()
        io = None
        connection.send(("closed",connection.stalls,deliveries))
    except BaseException as exc:
        connection.send(("error",f"{type(exc).__name__}: {exc}"))
    finally:
        if io is not None:
            io.close()
        if wake is not None:
            wake.close()
        connection.close()


class ProcessCommandIO:
    """Own communication GIL; only its helper's main thread writes the pipe."""
    def __init__(self, count, calibration, command_bound, cpus=None, *, allow_late=False):
        context = multiprocessing.get_context("spawn")
        self.connection,child = context.Pipe()
        self.connection = IOConnection(self.connection)
        self.process = context.Process(target=io_worker,args=(child,count,calibration,command_bound,cpus,allow_late),daemon=True)
        self.futures,self.samples,self.rows = {},{},{}
        self.closed,self.helper_stalls = False,[]
        self.delivery_records=[]
        self.process.start()
        child.close()
        if not self.connection.poll(30.):
            self.close()
            raise RuntimeError("communication helper initialization timed out")
        message = self.connection.recv()
        if message[0] != "ready":
            self.close()
            raise RuntimeError(f"communication helper initialization failed: {message}")
        self.placement = message[1]

    def _submit(self, kind, row, action=None):
        v = row["vehicle"]
        if v in self.futures or v in self.samples:
            raise ValueError("one I/O operation per control loop")
        target = self.samples if kind == "sample" else self.futures
        target[v] = Future()
        self.rows[row["sample_id"]] = row
        row["sample_submitted_s" if kind == "sample" else "io_submitted_s"] = time.perf_counter()
        self.connection.send(("sample",row) if kind == "sample" else ("send",action,row))

    def send(self, action, row):
        self._submit("send",row,action)

    def sample(self, row):
        self._submit("sample",row)

    def _receive(self, message, read_started=None):
        kind = message[0]
        if kind == "error":
            raise RuntimeError(f"communication helper: {message[1]}")
        if kind == "closed":
            self.helper_stalls = message[1]
            self.delivery_records=message[2] if len(message)>2 else []
            self.closed = True
            return
        if kind not in ("update","sample_done","command_done"):
            raise ValueError("unknown communication helper reply")
        result = message[-1]
        row = self.rows[result["sample_id"]]
        # Snapshot replies must not erase the parent's frozen model/deadline.
        if result["deadline_s"] != row["deadline_s"] or result["model"] != row["model"]:
            raise RuntimeError("communication helper changed a committed job")
        row.update(result)
        received = time.perf_counter()
        if result.get("status") == "applied":
            row.setdefault("command_response_received_s",received)
        if kind != "update":
            row["sample_response_received_s" if kind == "sample_done" else "io_response_received_s"] = received
            row["sample_parent_read_s" if kind=="sample_done" else "command_parent_read_s"]=read_started if read_started is not None else received
            target = self.samples if kind == "sample_done" else self.futures
            target[row["vehicle"]].set_result((message[1],row))

    def _drain(self, target):
        while self.connection.poll():
            began=time.perf_counter()
            self._receive(self.connection.recv(),began)
        ready = []
        for v,future in list(target.items()):
            if future.done():
                packet,row = future.result()
                ready.append((packet,row))
                del target[v]
                del self.rows[row["sample_id"]]
        return ready

    def drain_samples(self):
        return self._drain(self.samples)

    def drain(self):
        return self._drain(self.futures)

    def close(self):
        try:
            if not self.closed:
                self.connection.send(("stop",))
                end = time.perf_counter()+10.
                while not self.closed and time.perf_counter()<end:
                    if self.connection.poll(.05):
                        message = self.connection.recv()
                        self._receive(message)
                self.process.join(2.)
        except (EOFError,BrokenPipeError,OSError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()  # only this object's private helper
                self.process.join(2.)
            self.connection.close()
            self.closed = True
