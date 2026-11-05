"""One shared segmented CPU executor; frontend and inference may overlap."""

import gc
from contextlib import nullcontext
import multiprocessing
import time

from .framework_runtime import ControllerPool
from .framework_cpu_affinity import set_process_cpus
from .framework_runtime_wire import RuntimeConnection
from .safety_comparison_core import ProcessSegmentServer


def serve(connection, pool, policy, allow_late=False):
    """Receive/reselect only between complete segments on this process's core."""
    pending,segments = [],[]
    stopping = False
    while not stopping:
        # With no runnable work, read the task itself. A preliminary blocking
        # poll on Windows installs/cancels a separate zero-byte overlapped read.
        # Busy workers still inspect arrivals only between complete segments.
        first = connection.recv() if not pending else None
        while first is not None or connection.poll():
            message = first if first is not None else connection.recv()
            first = None
            if message[0] == "stop":
                stopping = True
                break
            _,feature,row = message
            job = pool.start(row["model"],feature)
            row["worker_enqueue_s"] = time.perf_counter()
            pending.append((job,row))
            connection.send(("accepted",row["sample_id"],row["model"],row["deadline_s"],row["worker_enqueue_s"]))
        if stopping:
            break
        item = min(pending,key=lambda item:((item[1]["priority"] if policy=="static" else item[1]["deadline_s"]),item[1]["enqueue_s"]))
        pending.remove(item)
        job,row = item
        start = time.perf_counter()
        if start >= row["deadline_s"] and not allow_late:
            row.update(status="expired_before_segment",ready_s=start)
            connection.send(("done",None,row))
            continue
        row.setdefault("first_service_s",start)
        before = job.cursor
        job.step()
        end = time.perf_counter()
        segments.append({"vehicle":row["vehicle"],"sample_id":row["sample_id"],
            "start_s":start,"end_s":end,"layer_start":before,"layer_end":job.cursor})
        row["compute_ms"] = row.get("compute_ms",0.)+1000*(end-start)
        if job.done:
            row.update(status="ready",ready_s=end)
            connection.send(("done",job.output,row))
        else:
            pending.append(item)
    return segments


def worker(connection, policy, cpus=None, allow_late=False, resident_worker=False,model_names=None,host_controlled=False):
    connection = RuntimeConnection(connection)
    try:
        placement = set_process_cpus(cpus) if cpus else {"enabled":False}
        pool = ControllerPool(model_names=model_names)
        pool.warm_worker()
        gc.collect()
        gc.disable()  # private, bounded-lifetime inference helper; models remain frozen
        if resident_worker:
            from .windows_worker_residency import WorkerResidency
        guard = WorkerResidency() if resident_worker else nullcontext()
        with guard as prepared:
            if prepared is not None:
                pool.warm_worker()  # page/working-set preparation must precede the last warmup
                placement["runtime_residency"] = prepared.report
            connection.send(("ready",pool.profile,pool.provenance(),placement))
            if host_controlled:
                from .framework_cpu_xpu import serve_segments
                segments=serve_segments(connection,pool,allow_late)
            else:segments = serve(connection,pool,policy,allow_late)
        connection.send(("closed",segments,connection.stalls))
    except BaseException as exc:
        connection.send(("error",f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class RuntimeServer(ProcessSegmentServer):
    def __init__(self, policy, cpus=None, *, allow_late=False, resident_worker=False,model_names=None,scheduler_cpus=None):
        context = multiprocessing.get_context("spawn")
        self.connection,child = context.Pipe()
        self.connection = RuntimeConnection(self.connection)
        if scheduler_cpus is None:
            self.process=context.Process(target=worker,args=(child,policy,cpus,allow_late,resident_worker,model_names),daemon=True)
        else:
            from .framework_cpu_xpu import coordinator_worker
            # A non-daemon CPU coordinator may own its private XPU process.
            self.process=context.Process(target=coordinator_worker,
                args=(child,policy,cpus,scheduler_cpus,allow_late,resident_worker,model_names))
        self.rows,self.finished,self.segments = {},[],[]
        self.error,self.closed,self.helper_stalls = None,False,[]
        self.cpu_dispatches=[]
        self.process.start()
        child.close()
        if not self.connection.poll(30.):
            self.close()
            raise RuntimeError("framework executor initialization timed out")
        message = self.connection.recv()
        if message[0] != "ready":
            self.close()
            raise RuntimeError(f"framework executor initialization failed: {message}")
        self.profile,self.provenance,self.placement = message[1:]

    def _receive(self, message):
        if message[0] == "closed":
            self.helper_stalls = message[2]
            if len(message)>3:self.cpu_dispatches=message[3]
        if message[0] != "accepted":
            return super()._receive(message)
        _,ident,model,deadline,enqueued = message
        row = self.rows[ident]
        if row["model"] != model or row["deadline_s"] != deadline:
            raise RuntimeError("executor acknowledgement changed a committed job")
        row["worker_enqueue_s"] = enqueued
        row["worker_accept_received_s"] = time.perf_counter()
