"""Shared-CPU multi-UAV safety comparison; empirical deadlines, not certificates."""

from collections import deque
import threading
import time
import multiprocessing

import numpy as np

from .decision_dataset import sha256
from .dynamic_probe_common import DATA, PLAIN
from .eval_offline import load_checkpoint, load_normalization


MODELS = ("plain_mlp_h9_w512_seed0", "plain_mlp_h21_w512_seed0", "plain_mlp_h9_w1024_seed0")
PERIODS_MS = (25., 50., 75., 100., 125., 150.)
# Narrow-gate pilot observed contact about0.48m laterally from the pawn centre.
# This0.50m envelope is a prototype assumption, not a certified attitude-wide bound.
BODY_RADIUS = .50
CLEARANCE = .025


def require_startup_vehicles(client, vehicles):
    """Do not fill a startup-sized RPC thread pool with runtime-added control tasks."""
    import json
    settings = json.loads(client.getSettingsString())
    configured = settings.get("Vehicles", {})
    missing = sorted(set(vehicles)-set(configured))
    if missing:
        raise RuntimeError(f"vehicles must exist in startup settings, not simAddVehicle: {missing}; "
            "restart Play with controller_architecture_study/configs/airsim_safety_12.json")
    missing = sorted(set(vehicles)-set(client.listVehicles()))
    if missing:
        raise RuntimeError(f"configured vehicles are not available in this Play session: {missing}")


class LayerPool:
    """Exact existing plain networks, paused only between FC/ReLU layer groups."""
    def __init__(self):
        import torch
        torch.set_num_threads(1)
        if not torch.set_flush_denormal(True):
            raise RuntimeError("CPU flush-denormal mode is unavailable")
        self.torch = torch
        self.norm = load_normalization(DATA)
        self.models = {name: load_checkpoint(PLAIN/name/"best_model.pt", device="cpu") for name in MODELS}
        self.layers = {name: tuple(model.children()) for name, model in self.models.items()}
        self.chunks = {name: tuple(torch.jit.script(torch.nn.Sequential(*layers[i:i+4]))
            for i in range(0, len(layers), 4)) for name, layers in self.layers.items()}
        self.profile = {}
        for name in MODELS:
            samples, segments = [], []
            for k in range(25):
                job = self.start(name, self.norm["x_mean"])
                started = time.perf_counter()
                while not job.done:
                    before = time.perf_counter()
                    job.step()
                    if k >= 5:
                        segments.append(1000*(time.perf_counter()-before))
                if k >= 5:
                    samples.append(1000*(time.perf_counter()-started))
            self.profile[name] = {"mean_ms": float(np.mean(samples)), "max_observed_ms": max(samples),
                "max_segment_ms": max(segments), "profile_samples": len(samples), "not_WCET": True}

    def start(self, name, feature):
        return LayerJob(self, name, feature)

    def warm_worker(self):
        self.torch.set_num_threads(1)
        self.torch.set_flush_denormal(True)
        for name in MODELS:
            for _ in range(3):
                job = self.start(name, self.norm["x_mean"])
                while not job.done:
                    job.step()

    def provenance(self):
        return {"input_contract": "decision_convex_local_relative_pv_237d_v1", "device": "cpu",
            "torch": str(self.torch.__version__), "normalization_sha256": sha256(DATA/"normalization.json"),
            "cpu_flush_denormal": True,
            "models": {n: {"checkpoint": str(PLAIN/n/"best_model.pt"),
                "sha256": sha256(PLAIN/n/"best_model.pt")} for n in MODELS},
            "segment": "up to2 existing FC layers including following ReLUs; no weight copying or retraining"}


class LayerJob:
    def __init__(self, pool, name, feature):
        self.pool, self.name, self.cursor = pool, name, 0
        x = np.asarray(feature, dtype=np.float32)
        if x.shape != (237,) or not np.isfinite(x).all():
            raise ValueError("one finite raw237D input required")
        self.h = pool.torch.from_numpy(((x-pool.norm["x_mean"])/pool.norm["x_std"])[None].copy())
        self.done, self.output = False, None

    def step(self):
        if self.done:
            raise ValueError("completed context cannot be executed again")
        layers = self.pool.layers[self.name]
        with self.pool.torch.inference_mode():
            end = min(self.cursor+4, len(layers))
            self.h = self.pool.chunks[self.name][self.cursor//4](self.h)
            self.cursor = end
            self.done = end == len(layers)
            if self.done:
                self.output = self.h.numpy()[0]*self.pool.norm["u_std"]+self.pool.norm["u_mean"]
        return self.done


class SegmentServer:
    """One real worker; same segment granularity for static FP and dynamic EDF."""
    def __init__(self, pool, policy):
        self.pool, self.policy = pool, policy
        self.condition = threading.Condition()
        self.pending, self.finished = [], deque()
        self.active, self.closed, self.error = None, False, None
        self.segments = []
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.work, name="safety-shared-CPU", daemon=True)
        self.thread.start()
        if not self.ready.wait(30.) or self.error:
            raise RuntimeError(self.error or "shared worker initialization timed out")

    def submit(self, feature, row):
        job = self.pool.start(row["model"], feature)
        row.setdefault("enqueue_s", time.perf_counter())
        row["worker_enqueue_s"] = time.perf_counter()
        with self.condition:
            self.pending.append((job, row))
            self.condition.notify()

    def key(self, item):
        row = item[1]
        return ((row["priority"] if self.policy == "static" else row["deadline_s"]), row["enqueue_s"])

    def work(self):
        try:
            self.pool.warm_worker()
            self.ready.set()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.closed)
                    if self.closed:
                        return
                    item = min(self.pending, key=self.key)
                    self.pending.remove(item)
                    self.active = item
                job, row = item
                start = time.perf_counter()
                if start >= row["deadline_s"]:
                    row.update(status="expired_before_segment", ready_s=start)
                    job.done = True
                else:
                    row.setdefault("first_service_s", start)
                    before = job.cursor
                    job.step()
                    end = time.perf_counter()
                    self.segments.append({"vehicle": row["vehicle"], "sample_id": row["sample_id"],
                        "start_s": start, "end_s": end, "layer_start": before, "layer_end": job.cursor})
                    row["compute_ms"] = row.get("compute_ms", 0.)+1000*(end-start)
                    if job.done:
                        row.update(status="ready", ready_s=end)
                with self.condition:
                    self.active = None
                    if job.done:
                        self.finished.append((job.output, row))
                    else:
                        self.pending.append(item)
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()

    def drain(self):
        with self.condition:
            values = list(self.finished)
            self.finished.clear()
        if self.error:
            raise RuntimeError(self.error)
        return values

    def forecast(self):
        with self.condition:
            rows = [row for _, row in self.pending]+([self.active[1]] if self.active else [])
        return rows

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()


def _process_worker(connection, policy):
    server = None
    try:
        pool = LayerPool()
        server = SegmentServer(pool, policy)
        connection.send(("ready", pool.profile, pool.provenance()))
        stopping = False
        while not stopping:
            while connection.poll():
                message = connection.recv()
                if message[0] == "stop":
                    stopping = True
                    break
                _, feature, row = message
                server.submit(feature, row)
            for action, row in server.drain():
                connection.send(("done", action, row))
            if not stopping:
                time.sleep(.0005)
        server.close()
        connection.send(("closed", server.segments))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if server is not None:
            server.close()
        connection.close()


class ProcessSegmentServer:
    """One shared inference worker, isolated from the AirSim frontend's Python GIL."""
    def __init__(self, policy):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_process_worker, args=(child, policy), daemon=True)
        self.rows, self.finished, self.segments = {}, [], []
        self.error, self.closed = None, False
        self.process.start()
        child.close()
        if not self.connection.poll(30.):
            self.close()
            raise RuntimeError("inference process initialization timed out")
        message = self.connection.recv()
        if message[0] != "ready":
            self.close()
            raise RuntimeError(f"inference process initialization failed: {message}")
        self.profile, self.provenance = message[1:]

    def submit(self, feature, row):
        row["enqueue_s"] = time.perf_counter()
        self.rows[row["sample_id"]] = row
        self.connection.send(("submit", feature, row))

    def _receive(self, message):
        if message[0] == "done":
            _, action, result = message
            row = self.rows.pop(result["sample_id"])
            row.update(result)
            self.finished.append((action, row))
        elif message[0] == "closed":
            self.segments, self.closed = message[1], True
        elif message[0] == "error":
            self.error = message[1]

    def drain(self):
        while self.connection.poll():
            self._receive(self.connection.recv())
        if self.error:
            raise RuntimeError(self.error)
        values, self.finished = self.finished, []
        return values

    def forecast(self):
        return list(self.rows.values())

    def close(self):
        if not self.closed:
            try:
                self.connection.send(("stop",))
                until = time.perf_counter()+10.
                while not self.closed and not self.error and time.perf_counter() < until:
                    if self.connection.poll(.05):
                        self._receive(self.connection.recv())
            except (EOFError, BrokenPipeError, OSError):
                pass
            self.process.join(2.)
            if self.process.is_alive():
                # Only this object's private compute helper, never the user's editor.
                self.process.terminate()
                self.process.join(2.)
                self.error = self.error or "inference helper did not stop normally"
            self.connection.close()
            self.closed = True


def safe_horizon(position, velocity, gates, net_width):
    """Prototype rate envelope for entering a gate's inflated frame; no CBF claim."""
    from gate_route import _gate_local
    candidates = []
    for gate in gates:
        p = _gate_local(np.asarray(position), gate)
        slab = BODY_RADIUS+CLEARANCE+.15/2
        if -3. <= p[0] <= slab:
            v = _gate_local(gate.center+np.asarray(velocity), gate)
            margin = np.array([net_width/2, 2.])-BODY_RADIUS-CLEARANCE-abs(p[1:])
            # Explicit prototype rate envelope, not the DNN approximation error.
            bound = np.maximum(abs(v[1:]), [2.5, 1.])
            # Being outside the aperture while still approaching is not contact.
            # A frame threat requires both longitudinal entry and transverse exit.
            entry = max(0.,abs(p[0])-slab)/max(abs(v[0]),5.)
            exit = float(np.min(np.maximum(margin,0.)/bound))
            candidates.append({"gate": gate.index, "margin_m": margin.tolist(),
                "longitudinal_entry_s":float(entry),"transverse_exit_s":exit,
                "horizon_s": max(float(entry),exit), "local_position": p.tolist()})
    horizon = min((c["horizon_s"] for c in candidates), default=.150)
    return min(.150, horizon), candidates


def bounded_release(now, deadline, service_bound, committed, own_vehicle, response_reserve=0.):
    """At most3 starts; EDF simulation of already committed one-shot jobs only."""
    candidates = sorted({max(now, deadline-service_bound-response_reserve-.003),
                         max(now, deadline-service_bound-response_reserve-.010), now}, reverse=True)
    for checked, release in enumerate(candidates, 1):
        jobs = [(r.get("planned_release_s", now), r["deadline_s"], r["service_bound_s"])
                for r in committed if r["vehicle"] != own_vehicle]
        jobs.append((release, deadline, service_bound))
        clock, feasible = now, True
        while jobs:
            ready = [j for j in jobs if j[0] <= clock]
            if not ready:
                clock = min(j[0] for j in jobs)
                ready = [j for j in jobs if j[0] <= clock]
            selected = min(ready, key=lambda j: j[1])
            jobs.remove(selected)
            clock += selected[2]
            if clock > selected[1]:
                feasible = False
                break
        if feasible:
            return release, True, checked
    return now, False, len(candidates)


def full_job_cost_profile(requests):
    """Predictive p95 envelopes of exclusive full jobs, explicitly not WCET bounds."""
    grouped = {}
    for row in requests:
        if row["status"] == "applied":
            duration = row["rpc_return_s"]-row["job_started_s"]
            if duration <= 0.:
                raise ValueError("full job duration must be positive")
            grouped.setdefault(row["model"], []).append(duration)
    return {name: {"samples": len(values), "p95_s": float(np.percentile(values, 95)),
        "max_observed_s": float(max(values)),
        "estimated_cost_s": float(max(.005, 1.25*np.percentile(values, 95)+.002)),
        "is_WCET": False} for name, values in grouped.items()}


def plan_after_delivery(row, observation, committed):
    """The new clock is the delivery RPC return, never the pre-RPC send time."""
    now = row["rpc_return_s"]
    deadline = min(observation["sample_s"]+observation["budget_s"], now+.150)
    release, feasible, checked = bounded_release(now, deadline, row["service_bound_s"],
        committed, row["vehicle"])
    slot = {"planned_release_s": release, "deadline_s": deadline, "vehicle": row["vehicle"],
        "service_bound_s": row["service_bound_s"]}
    decision = {"vehicle": row["vehicle"], "sample_s": observation["sample_s"],
        "budget_s": observation["budget_s"], "planning_time_s": now, "deadline_s": deadline,
        "release_s": release, "estimated_feasible": feasible, "candidate_limit": 3,
        "candidate_count": checked, "cost_contract": "exclusive full job including frontend, IPC, inference and command RPC"}
    return slot, decision


def joint_release_plan(now, slots):
    """Move all unreleased jobs together; absolute physical deadlines never move."""
    ordered = sorted(slots, key=lambda v: (slots[v]["deadline_s"], v))
    cumulative, slack = 0., float("inf")
    for vehicle in ordered:
        cumulative += slots[vehicle]["service_bound_s"]
        slack = min(slack, slots[vehicle]["deadline_s"]-now-cumulative)
    idle = max(0., slack-.010) if ordered else 0.
    clock, result = now+idle, {}
    for vehicle in ordered:
        result[vehicle] = {**slots[vehicle], "planned_release_s": clock}
        clock += slots[vehicle]["service_bound_s"]
    return result, {"planning_time_s": now, "estimated_feasible": slack >= 0.,
        "minimum_slack_s": slack if ordered else None, "collective_idle_s": idle,
        "edf_order": ordered, "candidate_count": 1, "candidate_limit": 1,
        "overload_action": "work-conserving EDF, no deadline extension; not an admission certificate"}


def paper_score(trace, duration):
    """Eq4-form on common20ms grid, Wx=I, K20 windows; failed/partial is missing."""
    if not trace or trace[0]["t"] > .025 or trace[-1]["t"] < duration:
        return None
    grid = np.arange(0., duration+1e-8, .02)
    values = np.interp(grid, [r["t"] for r in trace], [r["state_cost"] for r in trace])
    count = len(values)//21
    cost = float(values[:count*21].reshape(count, 21).sum(axis=1).mean())
    return {"J": 1/cost if cost > 0 else None, "mean_K20_cost": cost,
        "Wx_diagonal": [1.]*10, "grid_s": .02, "windows": count, "discarded_grid_samples": len(values)-count*21}
