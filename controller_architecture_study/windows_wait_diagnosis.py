"""Diagnostic-only Windows wait-chain sampling on a separate allowed CPU."""
import ctypes
from ctypes import wintypes as w
import threading
import time

import psutil

from .decision_dataset import atomic_json
from .framework_cpu_affinity import set_thread_cpus


class LockInfo(ctypes.Structure):
    _fields_=[("name",w.WCHAR*128),("timeout",ctypes.c_longlong),("alertable",w.BOOL)]


class ThreadInfo(ctypes.Structure):
    _fields_=[("pid",w.DWORD),("tid",w.DWORD),("wait_ms",w.DWORD),("switches",w.DWORD)]


class ObjectInfo(ctypes.Union):
    _fields_=[("lock",LockInfo),("thread",ThreadInfo)]


class Node(ctypes.Structure):
    _fields_=[("kind",w.DWORD),("status",w.DWORD),("info",ObjectInfo)]


class WaitChainProbe:
    def __init__(self,tid,path,cpu=16):
        self.tid,self.path,self.cpu=tid,path,cpu
        self.stop=threading.Event();self.samples=[];self.error=None

    def start(self):
        process=psutil.Process();self.previous=process.cpu_affinity()
        process.cpu_affinity(sorted(set(self.previous+[self.cpu])))
        self.thread=threading.Thread(target=self.run,name="wait-chain-diagnosis",daemon=True)
        self.thread.start()

    def run(self):
        api=ctypes.WinDLL("advapi32",use_last_error=True)
        api.OpenThreadWaitChainSession.argtypes=(w.DWORD,ctypes.c_void_p)
        api.OpenThreadWaitChainSession.restype=w.HANDLE
        api.GetThreadWaitChain.argtypes=(w.HANDLE,ctypes.c_size_t,w.DWORD,w.DWORD,
            ctypes.POINTER(w.DWORD),ctypes.POINTER(Node),ctypes.POINTER(w.BOOL))
        api.CloseThreadWaitChainSession.argtypes=(w.HANDLE,)
        session=None
        try:
            self.placement=set_thread_cpus([self.cpu])
            session=api.OpenThreadWaitChainSession(0,None)
            if not session:raise ctypes.WinError(ctypes.get_last_error())
            while not self.stop.is_set():
                nodes=(Node*16)();count=w.DWORD(16);cycle=w.BOOL();start=time.perf_counter()
                if not api.GetThreadWaitChain(session,0,0,self.tid,ctypes.byref(count),nodes,ctypes.byref(cycle)):
                    raise ctypes.WinError(ctypes.get_last_error())
                result=[]
                for n in nodes[:min(count.value,16)]:
                    row=dict(kind=n.kind,status=n.status)
                    if n.kind==8:
                        row.update(pid=n.info.thread.pid,tid=n.info.thread.tid,wait_ms=n.info.thread.wait_ms,
                            context_switches=n.info.thread.switches)
                    else:row["object_name"]=n.info.lock.name
                    result.append(row)
                self.samples.append(dict(start_s=start,end_s=time.perf_counter(),nodes=result,cycle=bool(cycle.value)))
                self.stop.wait(.010)
        except Exception as exc:self.error=f"{type(exc).__name__}: {exc}"
        finally:
            if session:api.CloseThreadWaitChainSession(session)

    def close(self):
        self.stop.set();self.thread.join(3.)
        psutil.Process().cpu_affinity(self.previous)
        atomic_json(self.path,dict(diagnostic_only=True,target_tid=self.tid,error=self.error,
            thread_stopped=not self.thread.is_alive(),samples=self.samples,
            caveat="WCT can report one running node for unsupported waits; not a complete context-switch trace"))
