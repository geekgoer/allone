"""Read-only 237D model provenance and a bounded, single-server inference queue."""

from concurrent.futures import Future
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import queue
import threading
import time

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256
from .convex_features import CONTRACT
from .eval_offline import load_checkpoint, load_normalization


RESULTS = ROOT / "controller_architecture_study/results"
OUTPUT = RESULTS / "dynamic_complexity_237d_v1"
BASE = RESULTS / "decision_convex_96k_v1"
DATA = RESULTS / "decision_convex_recovery_v1/datasets/features_recovery"
PLAIN = RESULTS / "decision_convex_recovery_scenic_falcon18_v1/runs"
RESIDUAL = RESULTS / "decision_convex_residual4_v1/runs"
RUNS = {
    "plain4x256": PLAIN / "plain_mlp_h4_w256_seed0",
    "plain4x512": PLAIN / "plain_mlp_h4_w512_seed0",
    "plain6x1024": PLAIN / "plain_mlp_h6_w1024_seed0",
    "residual21x1024": RESIDUAL / "residual_mlp_h21_w1024_seed0",
}
DEFAULT_MODELS = tuple(RUNS)  # Preserve the original four-model experiment.
RUNS["residual33x1024"] = RESIDUAL / "residual_mlp_h33_w1024_seed0"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def provenance(names, *, validation=False):
    """No Test values, no incomplete runs, and no unpaired normalization."""
    from .convex_recovery import verify_arm

    manifest = verify_arm(DATA, BASE, full=False)
    expected = manifest["artifact_sha256"]
    if validation and sha256(DATA / "validation.npz") != expected["validation.npz"]:
        raise ValueError("validation file changed")
    entries = {}
    for name in names:
        run = RUNS[name]
        cfg = read_json(run / "config.json")
        complete = read_json(run / "completion.json")
        if (cfg["input_dim"] != 237 or cfg["input_contract"] != CONTRACT
                or cfg["dataset_source_sha256"] != manifest["source_manifest_sha256"]
                or cfg["augmentation_manifest_sha256"] != manifest["augmentation_manifest_sha256"]
                or cfg["dataset_file_sha256"]["normalization.json"] != expected["normalization.json"]
                or cfg["dataset_file_sha256"]["validation.npz"] != expected["validation.npz"]
                or complete["epochs_completed"] < cfg["min_epochs"]):
            raise ValueError(f"incomplete or incompatible model: {name}")
        entries[name] = {"checkpoint": str(run / "best_model.pt"),
                         "checkpoint_sha256": sha256(run / "best_model.pt"),
                         "config_sha256": sha256(run / "config.json"),
                         "completion": complete}
    return {"input_contract": CONTRACT, "data_dir": str(DATA), "test_read": False,
            "normalization_sha256": expected["normalization.json"],
            "validation_sha256": expected["validation.npz"], "models": entries}


def new_output(kind):
    path = OUTPUT / (kind + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    path.mkdir(parents=True, exist_ok=False)
    return path


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max_observed": float(a.max())}


class ModelPool:
    """Only network weights reside on the device; outputs return to CPU before timing ends."""
    def __init__(self, names, device):
        import torch

        torch.set_num_threads(1)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU substitution")
        self.torch, self.device = torch, device
        self.norm = load_normalization(DATA)
        self.models = {n: load_checkpoint(RUNS[n] / "best_model.pt", device=device) for n in names}
        self.parameter_bytes = sum(p.numel()*p.element_size() for m in self.models.values() for p in m.parameters())

    def runtime_info(self):
        torch = self.torch
        return {"python": platform.python_version(), "platform": platform.platform(),
                "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                "cpu_threads": torch.get_num_threads(), "dtype": "float32", "autocast": False,
                "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
                "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
                "process_cuda_allocated_bytes": torch.cuda.memory_allocated() if self.device == "cuda" else None,
                "process_cuda_reserved_bytes": torch.cuda.memory_reserved() if self.device == "cuda" else None}

    def predict(self, name, feature):
        torch = self.torch
        start = time.perf_counter()
        x = np.asarray(feature, dtype=np.float32)
        if x.shape != (237,) or not np.isfinite(x).all():
            raise ValueError("expected one finite physical 237D input")
        x = (x - self.norm["x_mean"]) / self.norm["x_std"]
        with torch.inference_mode():
            tensor = torch.from_numpy(x[None]).to(self.device)
            if self.device == "cuda":
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
            output = self.models[name](tensor)
            if self.device == "cuda":
                b.record()
            normalized = output.cpu().numpy()[0]
            if self.device == "cuda":
                b.synchronize()
                gpu_ms = float(a.elapsed_time(b))
            else:
                gpu_ms = None
        action = normalized * self.norm["u_std"] + self.norm["u_mean"]
        if not np.isfinite(action).all():
            raise RuntimeError("nonfinite model output")
        return action, {"service_ms": (time.perf_counter()-start)*1000, "gpu_compute_ms": gpu_ms}


class InferenceService:
    """One FCFS worker, at most one outstanding request per vehicle.

    Independent producers never silently build an unbounded stale-state backlog.
    Busy/skipped/error requests remain in the denominator of deadline statistics.
    """
    def __init__(self, predictor, warmup=None, *, feedback=False):
        self.predictor, self.warmup = predictor, warmup
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.pending, self.rows = set(), []
        self.recent = deque(maxlen=4096) if feedback else None
        self.pending_models = {}
        self.inference_error = None
        self.ready = threading.Event()
        self.startup_error = None
        self.worker = threading.Thread(target=self._work, name="shared-237d-inference", daemon=True)
        self.worker.start()
        self.ready.wait()
        if self.startup_error is not None:
            raise RuntimeError("inference warmup failed") from self.startup_error

    def submit(self, model, feature, row):
        row.update(model=model, enqueue_s=time.perf_counter())
        with self.lock:
            self.rows.append(row)
            if row["vehicle"] in self.pending:
                row["status"] = "dropped_busy"
                return None
            self.pending.add(row["vehicle"])
            if self.recent is not None:
                self.pending_models[row["vehicle"]] = model
        future = Future()
        self.jobs.put((model, feature.copy(), row, future))
        return future

    def record_skip(self, vehicle, index, scheduled, **metadata):
        with self.lock:
            self.rows.append({"vehicle": vehicle, "cycle": index, "scheduled_s": scheduled,
                              "status": "skipped_release", **metadata})

    def record_abort(self, row):
        """Retain a sampled request rejected before inference submission."""
        with self.lock:
            row["status"] = "policy_abort"
            self.rows.append(row)

    def feedback_snapshot(self, now):
        """Only a bounded completion window and current pending counts, never history.

        Completed feedback records contain scalar copies and are never mutated;
        copying their references under the lock is safe and inexpensive.
        """
        if self.recent is None:
            raise RuntimeError("feedback statistics were not enabled")
        with self.lock:
            while self.recent and self.recent[0]["ready_s"] < now-2.:
                self.recent.popleft()
            return {"recent": list(self.recent),
                    "pending_model_counts": dict(Counter(self.pending_models.values())),
                    "inference_error": self.inference_error,
                    "capacity": self.recent.maxlen}

    def _work(self):
        try:
            if self.warmup:
                self.warmup()
        except BaseException as exc:
            self.startup_error = exc
        finally:
            self.ready.set()
        if self.startup_error:
            return
        while True:
            job = self.jobs.get()
            if job is None:
                return
            model, feature, row, future = job
            failure = None
            try:
                row["start_s"] = time.perf_counter()
                action, timing = self.predictor(model, feature)
                row.update(timing, ready_s=time.perf_counter(), output=action.tolist(), status="ready")
            except BaseException as exc:
                row.update(status="inference_error", error=f"{type(exc).__name__}: {exc}")
                failure = exc
            finally:
                with self.lock:
                    self.pending.discard(row["vehicle"])
                    self.pending_models.pop(row["vehicle"], None)
                    if failure is not None:
                        self.inference_error = row["error"]
                    elif self.recent is not None:
                        self.recent.append({k: row[k] for k in ("model", "status", "ready_s",
                            "service_ms", "sample_start_s", "scheduled_s")})
            if failure is None:
                future.set_result(action)
            else:
                future.set_exception(failure)

    def close(self):
        self.jobs.put(None)
        self.worker.join()


def release_schedule(start, duration, period, stop, service, vehicle, phase=0.):
    """No catch-up bursts: account for elapsed releases, then sample a fresh state."""
    count = int(np.ceil(duration/period))
    for k in range(count):
        scheduled = start + phase + k*period
        if stop.wait(max(0., scheduled-time.perf_counter())):
            break
        if time.perf_counter() >= scheduled+period:
            service.record_skip(vehicle, k, scheduled)
            continue
        yield k, scheduled


def summarize_timing(rows, deadline_ms):
    def limit(row):
        value = row.get("deadline_ms", deadline_ms)
        if value is None:
            raise ValueError("dynamic timing requires each request's own deadline")
        return value
    ready = [r for r in rows if r.get("status") == "ready"]
    output_misses = sum(r.get("status") != "ready" or
                        (r["ready_s"]-r["scheduled_s"])*1000 > limit(r) for r in rows)
    applied = [r for r in rows if "dnn_dispatch_s" in r]
    mpc_applied = [r for r in rows if "mpc_dispatch_s" in r]
    fields = {"release_jitter_ms": [(r["sample_start_s"]-r["scheduled_s"])*1000 for r in ready],
              "queue_ms": [(r["start_s"]-r["enqueue_s"])*1000 for r in ready],
              "sample_to_output_ms": [(r["ready_s"]-r["sample_start_s"])*1000 for r in ready],
              "release_to_output_ms": [(r["ready_s"]-r["scheduled_s"])*1000 for r in ready],
              "service_ms": [r["service_ms"] for r in ready],
              "gpu_compute_ms": [r["gpu_compute_ms"] for r in ready if r.get("gpu_compute_ms") is not None],
              "sample_to_dnn_dispatch_ms": [(r["dnn_dispatch_s"]-r["sample_start_s"])*1000 for r in applied]}
    fields["sample_to_mpc_dispatch_ms"] = [(r["mpc_dispatch_s"]-r["sample_start_s"])*1000 for r in mpc_applied]
    # Include sampled-but-busy rows too; filtering to completed inference hides load.
    stages = {
        "state_rpc_ms": ("sample_start_s", "state_ready_s"),
        "collision_rpc_ms": ("collision_start_s", "collision_ready_s"),
        "preprocess_wait_ms": ("preprocess_requested_s", "preprocess_start_s"),
        "reference_ms": ("preprocess_start_s", "reference_ready_s"),
        "feature_ms": ("reference_ready_s", "feature_ready_s"),
        "online_selection_ms": ("joint_online_start_s", "joint_online_ready_s"),
        "sample_to_enqueue_ms": ("sample_start_s", "enqueue_s"),
        "mpc_solve_ms": ("mpc_start_s", "mpc_ready_s"),
        "command_submit_ms": ("command_start_s", "command_ready_s"),
    }
    fields.update({name: [(r[end]-r[start])*1000 for r in rows if start in r and end in r]
                   for name, (start, end) in stages.items()})
    return {"released": len(rows), "ready": len(ready), "dropped_busy": sum(r.get("status") == "dropped_busy" for r in rows),
            "skipped_release": sum(r.get("status") == "skipped_release" for r in rows),
            "inference_errors": sum(r.get("status") == "inference_error" for r in rows),
            "output_deadline_misses": output_misses,
            "output_deadline_miss_fraction": output_misses/len(rows) if rows else None,
            "dnn_commands_dispatched": len(applied),
            "control_deadline_aborts": sum(r.get("control_outcome") in ("deadline_abort", "late_output_not_applied") for r in rows),
            "policy_aborts": sum(r.get("status") == "policy_abort" for r in rows),
            "peer_aborts": sum(r.get("control_outcome") == "peer_abort" for r in rows),
            "mpc_commands_dispatched": len(mpc_applied),
            "mpc_dispatch_deadline_misses": sum((r["mpc_dispatch_s"]-r["scheduled_s"])*1000 > limit(r) for r in mpc_applied),
            "dnn_dispatch_deadline_misses": sum((r["dnn_dispatch_s"]-r["scheduled_s"])*1000 > limit(r) for r in applied),
            **{k: describe(v) for k, v in fields.items()}}


def save_timing(path, rows, deadline_ms, metadata):
    with (path / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["scheduled_s"], r["vehicle"])):
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    report = {**metadata, "timing_schema_version": 3 if deadline_ms is None else 2,
              "report_path": str(path / "report.json"),
              "deadline_ms": deadline_ms, "timing": summarize_timing(rows, deadline_ms),
              "per_vehicle": {v: summarize_timing([r for r in rows if r["vehicle"] == v], deadline_ms)
                              for v in sorted({r["vehicle"] for r in rows})},
              "limitations": ["Host monotonic clock; sample timestamp is RPC request start, not sensor acquisition.",
                              "Dispatch is an async RPC submission, not a physical actuation acknowledgment.",
                              "Observed quantiles are not WCET or safety certificates; no FPGA latency claim.",
                              "CUDA event interval can include device contention/launch gaps, not only active kernel work."]}
    if report["timing"]["inference_errors"]:
        report["status"] = "INCOMPLETE"
    timing = report["timing"]
    report["realtime_passed"] = bool(
        report.get("status") == "COMPLETE" and timing["released"] > 0
        and report.get("schedule_accounting_passed", True)
        and timing["released"] == report.get("planned_releases", timing["released"])
        and timing["output_deadline_misses"] == 0
        and timing["mpc_dispatch_deadline_misses"] == 0
        and timing["dnn_dispatch_deadline_misses"] == 0)
    if "all_15_crossings_passed" in report:
        report["flight_passed"] = report["all_15_crossings_passed"]
        report["acceptance_passed"] = bool(report["flight_passed"] and report["realtime_passed"]
            and report.get("clock_ratio_passed", True) and report.get("tracking_window_covered", True))
    atomic_json(path / "report.json", report)
    print(f"Report: {path / 'report.json'}", flush=True)
    return report
