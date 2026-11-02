"""Scoped preparation of our CPU inference helper; not FPGA timing or WCET."""

import ctypes
import os
import time

import psutil


class WorkerResidency:
    """Keep the three observed interpreter/Torch DLL images resident during a trial."""
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("worker residency requires Windows")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        signatures = {
            "GetModuleHandleW": ([ctypes.c_wchar_p], pointer),
            "VirtualQuery": ([pointer, pointer, size], size),
            "VirtualLock": ([pointer, size], ctypes.c_int),
            "VirtualUnlock": ([pointer, size], ctypes.c_int),
            "GetProcessWorkingSetSize": ([pointer, ctypes.POINTER(size), ctypes.POINTER(size)], ctypes.c_int),
            "SetProcessWorkingSetSize": ([pointer, size, size], ctypes.c_int),
            "GetThreadPriority": ([pointer], ctypes.c_int),
            "SetThreadPriority": ([pointer, ctypes.c_int], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.process = psutil.Process()
        self.old_priority = self.process.nice()
        self.old_thread_priority = self.api.GetThreadPriority(pointer(-2))
        self.old_min, self.old_max = size(), size()
        self._check(self.api.GetProcessWorkingSetSize(pointer(-1), ctypes.byref(self.old_min), ctypes.byref(self.old_max)))
        self.locked, self.report = [], {}

    @staticmethod
    def _check(ok):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self):
        class MemoryInfo(ctypes.Structure):
            _fields_ = [("base", ctypes.c_void_p), ("allocation", ctypes.c_void_p),
                ("allocation_protect", ctypes.c_ulong), ("partition", ctypes.c_ushort),
                ("size", ctypes.c_size_t), ("state", ctypes.c_ulong),
                ("protect", ctypes.c_ulong), ("kind", ctypes.c_ulong)]

        began = time.perf_counter()
        regions, modules = [], []
        for name in ("torch_cpu.dll", "torch_python.dll", "python311.dll"):
            base = self.api.GetModuleHandleW(name)
            if not base:
                raise RuntimeError(f"expected loaded runtime DLL: {name}")
            pe_offset = ctypes.c_uint32.from_address(base+0x3c).value
            image_size = ctypes.c_uint32.from_address(base+pe_offset+24+56).value
            cursor, readable = base, 0
            while cursor < base+image_size:
                info = MemoryInfo()
                self._check(self.api.VirtualQuery(cursor, ctypes.byref(info), ctypes.sizeof(info)))
                end = min(info.base+info.size, base+image_size)
                if info.state == 0x1000 and not info.protect & (0x01 | 0x100):
                    regions.append((cursor, end-cursor))
                    readable += end-cursor
                cursor = end
            modules.append(dict(name=name, image_bytes=image_size, locked_bytes=readable))
        total = sum(length for _, length in regions)
        if total > 384*1024**2:
            raise RuntimeError("runtime images exceed the 384MiB diagnostic pinning limit")
        try:
            minimum = max(self.old_min.value, total+32*1024**2)
            resident_before = self.process.memory_info().rss
            maximum = max(self.old_max.value, minimum+16*1024**2, resident_before+total+64*1024**2)
            self._check(self.api.SetProcessWorkingSetSize(ctypes.c_void_p(-1), minimum,
                maximum))
            for address, length in regions:
                self._check(self.api.VirtualLock(address, length))
                self.locked.append((address, length))
            # Only this private helper changes priority. No realtime/high class or other-app edits.
            self.process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            self._check(self.api.SetThreadPriority(ctypes.c_void_p(-2), 1))
            self.report = dict(enabled=True, modules=modules, locked_bytes=total,
                startup_ms=1000*(time.perf_counter()-began), priority_class=int(self.process.nice()),
                relative_thread_priority=self.api.GetThreadPriority(ctypes.c_void_p(-2)),
                working_set_before_bytes=resident_before, working_set_minimum_bytes=minimum,
                working_set_maximum_bytes=maximum,
                model_weights_locked=False, is_WCET=False, FPGA_timing_equivalent=False)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        for address, length in reversed(self.locked):
            self._check(self.api.VirtualUnlock(address, length))
        self.locked.clear()
        self._check(self.api.SetProcessWorkingSetSize(ctypes.c_void_p(-1), self.old_min.value, self.old_max.value))
        self._check(self.api.SetThreadPriority(ctypes.c_void_p(-2), self.old_thread_priority))
        self.process.nice(self.old_priority)

