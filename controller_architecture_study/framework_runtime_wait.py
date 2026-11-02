"""Windows release timer combined with IPC readiness; no global timer change."""

import ctypes
from multiprocessing.connection import wait
import os
import time


class CompletionEvent:
    """Auto-reset wakeup: callbacks signal it; only the manager writes its pipe."""
    def __init__(self):
        self.api=ctypes.WinDLL("kernel32",use_last_error=True)
        self.api.CreateEventW.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_wchar_p)
        self.api.CreateEventW.restype=ctypes.c_void_p
        self.api.SetEvent.argtypes=(ctypes.c_void_p,)
        self.api.CloseHandle.argtypes=(ctypes.c_void_p,)
        self.handle=self.api.CreateEventW(None,False,False,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def set(self):
        if not self.api.SetEvent(self.handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle=None


class DeadlineWaiter:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("this CPU prototype requires the Windows release timer")
        self.api = ctypes.WinDLL("kernel32",use_last_error=True)
        self.api.CreateWaitableTimerExW.argtypes = (ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong)
        self.api.CreateWaitableTimerExW.restype = ctypes.c_void_p
        self.api.SetWaitableTimerEx.argtypes = (ctypes.c_void_p,ctypes.POINTER(ctypes.c_longlong),ctypes.c_long,
            ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong)
        self.api.SetWaitableTimerEx.restype = ctypes.c_int
        self.api.CancelWaitableTimer.argtypes = (ctypes.c_void_p,)
        self.api.CloseHandle.argtypes = (ctypes.c_void_p,)
        # HIGH_RESOLUTION; SYNCHRONIZE | TIMER_MODIFY_STATE, private unnamed handle.
        self.handle = self.api.CreateWaitableTimerExW(None,None,2,0x00100002)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def wait(self, connections, wake_at):
        remaining = wake_at-time.perf_counter()
        if remaining <= 0.:
            return []
        due = ctypes.c_longlong(-max(1,int(remaining*1e7)))
        if not self.api.SetWaitableTimerEx(self.handle,ctypes.byref(due),0,None,None,None,0):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return wait([*connections,self.handle])
        finally:
            self.api.CancelWaitableTimer(self.handle)

    def close(self):
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self,*unused):
        self.close()
