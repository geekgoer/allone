"""Scoped Windows CPU placement for the CPU analogue, not real-time isolation."""

import ctypes
import os
import struct
import threading

import psutil


def cpu_topology():
    if os.name != "nt":
        raise RuntimeError("physical-core placement currently requires Windows CPU-set information")
    api = ctypes.WinDLL("kernel32",use_last_error=True).GetSystemCpuSetInformation
    size = ctypes.c_ulong()
    api(None,0,ctypes.byref(size),None,0)
    if not size.value:
        raise ctypes.WinError(ctypes.get_last_error())
    data = ctypes.create_string_buffer(size.value)
    if not api(data,size.value,ctypes.byref(size),None,0):
        raise ctypes.WinError(ctypes.get_last_error())
    rows,offset = [],0
    while offset < size.value:
        length,kind = struct.unpack_from("<II",data.raw,offset)
        if length < 8 or offset+length > size.value:
            raise RuntimeError("invalid Windows CPU-set record")
        if kind == 0:
            ident,group,logical,core,cache,numa,efficiency,flags = struct.unpack_from("<IHBBBBBB",data.raw,offset+8)
            rows.append({"id":ident,"group":group,"logical":logical,"core":core,
                "efficiency_class":efficiency,"flags":flags})
        offset += length
    return rows


def choose_layout(topology, allowed):
    """Four distinct fastest-class physical cores; never treat SMT as extra cores."""
    if not topology or any(r["group"] != 0 for r in topology):
        raise ValueError("this experiment supports one Windows processor group only")
    fastest = max(r["efficiency_class"] for r in topology)
    cores = {}
    for row in topology:
        if row["efficiency_class"] == fastest and row["logical"] in allowed:
            cores.setdefault(row["core"],[]).append(row["logical"])
    # Avoid core0 and retain other physical cores for Unreal/Windows.
    candidates = [c for c in sorted(cores) if c != 0]
    if len(candidates) < 5:
        raise ValueError("need at least five available nonzero fastest-class physical cores")
    io_a,io_b,frontend,inference = candidates[-4:]
    selected = {io_a,io_b,frontend,inference}
    reserved = sorted(r["logical"] for r in topology if r["core"] in selected)
    return {"inference_cpus":[min(cores[inference])],"frontend_cpus":[min(cores[frontend])],
        "io_cpus":[min(cores[io_a]),min(cores[io_b])],"reserved_physical_siblings":reserved,
        "parent_process_cpus":sorted([min(cores[c]) for c in (io_a,io_b,frontend)]),
        "efficiency_class":fastest}


def set_process_cpus(cpus):
    process = psutil.Process()
    previous = process.cpu_affinity()
    process.cpu_affinity(cpus)
    actual = process.cpu_affinity()
    if sorted(actual) != sorted(cpus):
        raise RuntimeError("process affinity readback mismatch")
    return {"pid":process.pid,"requested":list(cpus),"actual":actual,"previous":previous}


def set_thread_cpus(cpus):
    if os.name != "nt":
        raise RuntimeError("thread placement currently requires Windows")
    api = ctypes.WinDLL("kernel32",use_last_error=True)
    api.GetCurrentThread.restype = ctypes.c_void_p
    api.SetThreadAffinityMask.argtypes = (ctypes.c_void_p,ctypes.c_size_t)
    api.SetThreadAffinityMask.restype = ctypes.c_size_t
    class GroupAffinity(ctypes.Structure):
        _fields_ = [("mask",ctypes.c_size_t),("group",ctypes.c_ushort),("reserved",ctypes.c_ushort*3)]
    api.GetThreadGroupAffinity.argtypes = (ctypes.c_void_p,ctypes.POINTER(GroupAffinity))
    thread = api.GetCurrentThread()
    previous = api.SetThreadAffinityMask(thread,sum(1<<c for c in cpus))
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    result = GroupAffinity()
    if not api.GetThreadGroupAffinity(thread,ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    actual = [c for c in range(64) if result.mask & (1<<c)]
    if result.group != 0 or sorted(actual) != sorted(cpus):
        raise RuntimeError("thread affinity readback mismatch")
    return {"thread_id":threading.get_native_id(),"requested":list(cpus),"actual":actual,
        "previous":[c for c in range(64) if previous & (1<<c)]}


class CpuPlacement:
    """Restore our process and the exact 12-drone editor even after trial failure."""
    def __init__(self, enabled=True):
        self.enabled,self.editors,self.parent,self.thread = enabled,[],None,None
        self.report = {"enabled":enabled,"exclusive_os_reservation":False,"is_WCET":False,
            "gil_removed_from_frontend_io":True,"FPGA_timing_equivalent":False}

    def __enter__(self):
        if not self.enabled:
            return self
        topology = cpu_topology()
        self.layout = choose_layout(topology,psutil.Process().cpu_affinity())
        self.report.update(topology=topology,layout=self.layout,editors=self.editors)
        try:
            for process in psutil.process_iter(["name","cmdline"]):
                if ((process.info["name"] or "").lower() == "ue4editor.exe"
                        and any("airsim_safety_12.json" in a for a in (process.info["cmdline"] or []))):
                    old = process.cpu_affinity()
                    new = sorted(set(old)-set(self.layout["reserved_physical_siblings"]))
                    if not new:
                        raise RuntimeError("no editor CPUs remain after experiment-core exclusion")
                    row = {"pid":process.pid,"created_s":process.create_time(),"previous":old,"requested":new}
                    self.editors.append(row)
                    process.cpu_affinity(new)
                    row["actual"] = process.cpu_affinity()
                    if row["actual"] != new:
                        raise RuntimeError("editor affinity readback mismatch")
            # Capture the original thread mask before process restriction changes it.
            original_thread = set_thread_cpus(psutil.Process().cpu_affinity())["previous"]
            self.parent = set_process_cpus(self.layout["parent_process_cpus"])
            self.thread = set_thread_cpus(self.layout["frontend_cpus"])
            self.thread["previous"] = original_thread
            self.report.update(parent=self.parent,frontend_thread=self.thread)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *unused):
        failures = []
        if self.parent is not None:
            try:
                set_process_cpus(self.parent["previous"])
                if self.thread is not None:
                    set_thread_cpus(self.thread["previous"])
            except Exception as exc:
                failures.append(f"parent: {exc}")
        for row in self.editors:
            try:
                process = psutil.Process(row["pid"])
                if process.create_time() != row["created_s"]:
                    raise RuntimeError("PID reused; refuse to change a different process")
                process.cpu_affinity(row["previous"])
                row["restored"] = process.cpu_affinity() == row["previous"]
                if not row["restored"]:
                    failures.append(f"editor {row['pid']}: restore mismatch")
            except psutil.NoSuchProcess:
                row["exited_before_restore"] = True
            except Exception as exc:
                failures.append(f"editor {row['pid']}: {exc}")
        self.report.update(restore_errors=failures,restoration_passed=not failures)
        if failures:
            raise RuntimeError("CPU placement restoration failed: "+"; ".join(failures))
