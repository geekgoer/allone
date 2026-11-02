"""C ABI bridge: no Python I/O threads, process queues or per-frame JSON."""
import ctypes as C
from concurrent.futures import Future
from collections import deque
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from airsim_interface import RawDroneState
from control_adapter import control_to_airsim_command, throttle_to_thrust


LIBRARY = Path(__file__).with_name('native_runtime')/'build/Release/falcon_native_io.dll'


class NativeEvent(C.Structure):
    _fields_ = [('kind',C.c_int32),('vehicle',C.c_int32),('id',C.c_uint64),('version',C.c_uint64),
        *[(k,C.c_double) for k in ('submitted','started','state_end','collision_end','finished','planned','period')],
        ('sim_ns',C.c_uint64),('collision_ns',C.c_uint64),('state',C.c_double*16),('control',C.c_double*4),('command',C.c_double*4),
        ('collided',C.c_int32),('status',C.c_int32),('object',C.c_char*128),('error',C.c_char*256)]


def load_library(path=LIBRARY):
    if not path.is_file():
        raise RuntimeError(f'Build native runtime first: {path}')
    dll=C.CDLL(str(path))  # Releases the GIL around every C++ API call.
    handle,integer,token,number=C.c_void_p,C.c_int,C.c_uint64,C.c_double
    signatures={
        'error':([],C.c_char_p),'clock':([],number),'event_size':([],C.c_size_t),
        'create':([C.c_char_p,integer,C.c_char_p,token,C.POINTER(number),number,integer,integer,number],handle),
        'handle':([handle],C.c_size_t),'poll':([handle,C.POINTER(NativeEvent)],integer),
        'sample':([handle,integer,token],integer),
        'send':([handle,integer,token,C.POINTER(number),number],integer),
        'schedule':([handle,integer,token,token,number,number,number,C.POINTER(NativeEvent)],integer),
        'schedule_batch':([handle,C.POINTER(NativeEvent),integer],integer),
        'calendar_matches':([handle,C.POINTER(NativeEvent),integer],integer),
        'guarded_batch':([handle,C.POINTER(NativeEvent),integer,integer],integer),
        'cancel':([handle,integer],integer),'stop':([handle],None),'destroy':([handle],None)}
    for name,(args,result) in signatures.items():
        f=getattr(dll,'falcon_'+name);f.argtypes=args;f.restype=result
    if dll.falcon_event_size()!=C.sizeof(NativeEvent):raise RuntimeError('native ABI size mismatch')
    before=time.perf_counter();native=dll.falcon_clock();after=time.perf_counter()
    if not before-.0001<=native<=after+.0001:raise RuntimeError('native/Python clock epochs differ')
    return dll


class NativeCommandIO:
    def __init__(self, vehicles, calibration, command_bound, cpus=None, *, allow_late=False,
            host='127.0.0.1',port=41451,mock=False,mock_delay_s=0.):
        self.dll=load_library()
        self.vehicles=list(vehicles);self.calibration=calibration
        self.index={v:i for i,v in enumerate(self.vehicles)}
        calibration.validate()
        cal=(C.c_double*5)(calibration.hover_throttle,calibration.hover_thrust_accel,
            calibration.thrust_accel_per_throttle,calibration.min_throttle,calibration.max_throttle)
        self.handle=self.dll.falcon_create(host.encode(),port,'\n'.join(self.vehicles).encode(),
            sum(1<<i for i in (cpus or [])),cal,command_bound,allow_late,mock,mock_delay_s)
        if not self.handle:raise RuntimeError(self.dll.falcon_error().decode())
        self.connection=SimpleNamespace(connection=self.dll.falcon_handle(self.handle),stalls=[])
        self.futures,self.samples,self.rows={},{},{}
        self.tokens=0;self.closed=False
        self.releases=deque();self.templates={};self.scheduled={};self.scheduled_versions={};self.version=0
        self.delivery_records=[];self.helper_stalls=[]
        self.placement=dict(backend='cpp_airlib',requested=list(cpus or []),
            python_io_threads=0,python_ipc_hops=0,native_workers=len(vehicles),
            native_absolute_release_thread=True,exclusive_os_reservation=False,
            rpc_internal_threads='inherit the native owner thread CPU mask in the local build',library=str(LIBRARY))

    def check(self, result):
        if result<0:raise RuntimeError(self.dll.falcon_error().decode())
        return result

    def _submit(self, row, action=None):
        v=row['vehicle']
        if v in self.futures or v in self.samples:raise ValueError('one I/O operation per control loop')
        self.tokens+=1;token=self.tokens
        target=self.samples if action is None else self.futures
        if action is not None:
            action=np.asarray(action,dtype=float).reshape(4)
            if not np.isfinite(action).all():raise ValueError('nonfinite control')
            values=(C.c_double*4)(*action)
            self.check(self.dll.falcon_send(self.handle,self.index[v],token,values,row['deadline_s']))
        else:self.check(self.dll.falcon_sample(self.handle,self.index[v],token))
        row['sample_submitted_s' if action is None else 'io_submitted_s']=time.perf_counter()
        target[v]=Future();self.rows[token]=(row,target)

    def sample(self,row):self._submit(row)
    def send(self,action,row):self._submit(row,action)

    @staticmethod
    def packet(event):
        values=np.array(event.state,dtype=float)
        raw=RawDroneState(values[:3],values[3:7],values[7:10],values[10:13],values[13:16],int(event.sim_ns))
        hit=SimpleNamespace(time_stamp=int(event.collision_ns),has_collided=bool(event.collided),
            object_name=event.object.decode(errors='replace'))
        return raw,hit,event.planned

    def pump(self):
        event=NativeEvent()
        while self.check(self.dll.falcon_poll(self.handle,C.byref(event))):
            if event.kind==5:raise RuntimeError('native I/O: '+event.error.decode(errors='replace'))
            if event.kind==4:
                row=dict(self.templates[event.version])
                v=self.vehicles[event.vehicle]
                delta=event.planned-row['planned_release_s']
                row.update(sample_id=f'{v}:{event.id}',planned_release_s=event.planned,
                    minimum_release_s=event.planned,requested_period_s=event.period,
                    slot=int(event.id),native_release_s=event.finished,native_calendar_version=int(event.version),
                    status='planned',nominal_deadline_s=event.planned+event.period)
                # A held-control budget is absolute; carrying it forward must not renew it.
                row['deadline_s']=(row['deadline_s']+delta if row.get('native_static') else row['deadline_s'])
                row['previous_release_s']=event.planned-event.period
                self.releases.append(row)
                continue
            row,target=self.rows[event.id];v=row['vehicle']
            if event.kind in (2,3):
                row.update(io_submitted_s=event.submitted,command_helper_received_s=event.started,
                    io_started_s=event.started,command_send_s=event.started)
                row['status']=('applied','expired_not_applied','delivery_budget_exhausted','cancelled_before_command')[event.status]
                if event.status==0:
                    action=np.array(event.control)
                    command=control_to_airsim_command(action,self.calibration,duration=1.)
                    row.update(control=action.tolist(),airsim_command=vars(command),mapped_control=[
                        throttle_to_thrust(command.throttle,self.calibration),command.roll_rate,-command.pitch_rate,-command.yaw_rate])
                    row['native_rpc_arguments']=list(event.command)
                if event.kind==2:
                    row['rpc_return_s']=event.finished
                    row['control_ack_token']=int(event.id)
                    row['command_response_received_s']=time.perf_counter()
                    continue
                row.update(io_finished_s=event.finished,command_state_rpc_ms=1000*(event.state_end-event.planned),
                    command_collision_rpc_ms=1000*(event.collision_end-event.state_end))
            else:
                row.update(sample_submitted_s=event.submitted,sample_helper_received_s=event.started,
                    sample_io_started_s=event.started,sample_io_finished_s=event.finished,
                    state_rpc_ms=1000*(event.state_end-event.started),
                    collision_rpc_ms=1000*(event.collision_end-event.state_end))
            row['sample_response_received_s' if event.kind==1 else 'io_response_received_s']=time.perf_counter()
            packet=self.packet(event) if event.kind==1 or event.status==0 else None
            target[v].set_result((packet,row));del self.rows[event.id]
            self.delivery_records.append(dict(sample_id=row['sample_id'],kind='sample_done' if event.kind==1 else 'command_done',
                start_s=event.finished,end_s=time.perf_counter(),native=True))

    def _drain(self,target):
        self.pump();ready=[]
        for v,future in list(target.items()):
            if future.done():ready.append(future.result());del target[v]
        return ready

    def drain_samples(self):return self._drain(self.samples)
    def drain(self):return self._drain(self.futures)

    @staticmethod
    def plan_key(row):
        return tuple(row.get(k) for k in ('sample_id','model','planned_release_s','requested_period_s','deadline_s','priority'))

    def schedule(self,row,end,*,static=False):
        v=row['vehicle'];key=self.plan_key(row)
        if self.scheduled.get(v)==key:return
        self.version+=1
        sequence=int(row['sample_id'].rsplit(':',1)[1])
        installed=NativeEvent()
        self.check(self.dll.falcon_schedule(self.handle,self.index[v],sequence,self.version,
            row['planned_release_s'],row['requested_period_s'],end,C.byref(installed)))
        delta=installed.planned-row['planned_release_s']
        row.update(sample_id=f'{v}:{installed.id}',planned_release_s=installed.planned,
            nominal_deadline_s=installed.planned+row['requested_period_s'])
        if static:row['deadline_s']+=delta
        self.templates[self.version]={**row,'native_static':static}
        self.scheduled[v]=self.plan_key(row)
        self.scheduled_versions[v]=self.version

    def cancel(self,vehicle):
        self.check(self.dll.falcon_cancel(self.handle,self.index[vehicle]))
        self.scheduled.pop(vehicle,None)
        self.scheduled_versions.pop(vehicle,None)

    def calendar_matches(self,plans):
        """Read-only generation check under the native publication mutex."""
        events=(NativeEvent*len(plans))()
        for event,(v,row) in zip(events,plans.items()):
            if v not in self.scheduled_versions:return False
            event.vehicle=self.index[v];event.id=int(row['sample_id'].rsplit(':',1)[1])
            event.version=self.scheduled_versions[v]
            event.planned=row['planned_release_s'];event.period=row['requested_period_s']
        return self.check(self.dll.falcon_calendar_matches(self.handle,events,len(events)))==0

    def calendar_basis(self,plans):
        return {v:dict(sample_id=r['sample_id'],version=self.scheduled_versions[v],
            planned_release_s=r['planned_release_s'],requested_period_s=r['requested_period_s']) for v,r in plans.items()}

    def guarded_batch(self,expected,proposal,source_controls,end,expires_s,*,commit=False):
        """CAS calendar AND actual control context under ONE native mutex.

        Dry-run has no calendar/template/version side effects. Complete fresh
        pipeline admission and observation/certificate checks remain required.
        """
        if not expected or expected.keys()!=proposal.keys() or expected.keys()!=source_controls.keys():
            raise ValueError('guard requires every loop in the checked group')
        events=(NativeEvent*len(proposal))();version=self.version
        for e,(v,row) in zip(events,proposal.items()):
            old=expected[v]
            if row['sample_id']!=old['sample_id']:raise ValueError('published identity cannot be replaced by guard')
            version+=1;e.vehicle=self.index[v];e.id=int(old['sample_id'].rsplit(':',1)[1]);e.version=old['version']
            e.submitted=old['planned_release_s'];e.started=old['requested_period_s']
            e.planned=row['planned_release_s'];e.period=row['requested_period_s'];e.finished=end
            e.state_end=expires_s;e.collision_ns=source_controls[v];e.sim_ns=version
        status=self.check(self.dll.falcon_guarded_batch(self.handle,events,len(events),int(commit)))
        if status==0 and commit:
            self.version=version
            for e,(v,row) in zip(events,proposal.items()):
                self.templates[e.sim_ns]={**row,'native_static':False}
                self.scheduled[v]=self.plan_key(row);self.scheduled_versions[v]=int(e.sim_ns)
        return dict(passed=status==0,installed=status==0 and commit,status=status,
            reason={0:'context_current',1:'calendar_changed',2:'control_changed_or_inflight',3:'review_expired_or_release_late'}[status])

    def schedule_batch(self,plans,end):
        changed={v for v,row in plans.items() if self.scheduled.get(v)!=self.plan_key(row)}
        if not changed:return True
        # Validate the whole forecast snapshot, including loops whose rate is
        # unchanged. An unnoticed publication there also invalidates admission.
        rows=list(plans.values());events=(NativeEvent*len(rows))()
        for event,row in zip(events,rows):
            v=row['vehicle']
            if v in changed:self.version+=1
            event.vehicle=self.index[v]
            event.id=int(row['sample_id'].rsplit(':',1)[1])
            event.version=self.version if v in changed else self.scheduled_versions[v]
            event.planned=row['planned_release_s'];event.period=row['requested_period_s'];event.finished=end
        if self.check(self.dll.falcon_schedule_batch(self.handle,events,len(events))):return False
        for event,row in zip(events,rows):
            if row['vehicle'] in changed:
                self.templates[event.version]={**row,'native_static':False}
                self.scheduled[row['vehicle']]=self.plan_key(row)
                self.scheduled_versions[row['vehicle']]=event.version
        return True

    def drain_releases(self):
        self.pump();rows=list(self.releases);self.releases.clear();return rows

    def close(self):
        if not self.closed:
            self.dll.falcon_stop(self.handle)
            try:self.pump()
            finally:
                self.dll.falcon_destroy(self.handle);self.handle=None;self.closed=True
