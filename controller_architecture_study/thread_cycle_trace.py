"""Optional Windows per-thread cycles; these are not CPU milliseconds."""
import ctypes
import os

if os.name == 'nt':
    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel.GetCurrentThread.restype = ctypes.c_void_p
    _kernel.QueryThreadCycleTime.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong))
    _kernel.QueryThreadCycleTime.restype = ctypes.c_int


def cycles():
    if os.name != 'nt':
        return None
    value = ctypes.c_ulonglong()
    if not _kernel.QueryThreadCycleTime(_kernel.GetCurrentThread(), ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return value.value
