"""Register committed releases independently of pipeline dispatch capacity."""
from collections import deque


class ReleaseQueue:
    def __init__(self, vehicles, policy='all'):
        if policy not in ('all','latest'):raise ValueError('unknown release queue policy')
        self.policy=policy
        self.queues = {v: deque() for v in vehicles}
        self.superseded=[]

    def publish(self, row, now):
        if now < row['planned_release_s']:
            raise ValueError('cannot release a future job')
        row.update(job_released_s=now, status='released_waiting',
                   committed_release_s=row['planned_release_s'],
                   committed_model=row['model'], committed_deadline_s=row['deadline_s'])
        if 'previous_release_s' in row:
            row['selected_release_interval_s']=row['planned_release_s']-row['previous_release_s']
        queue=self.queues[row['vehicle']]
        if self.policy=='latest':
            while queue:
                old=queue.popleft()
                old.update(status='superseded_before_dispatch',superseded_by=row['sample_id'],discarded_s=now)
                self.superseded.append(old)
        queue.append(row)

    def dispatchable(self, pending, limit, now=None):
        if len(pending) >= limit:
            return {}
        return {v: q[0] for v, q in self.queues.items() if q and v not in pending
            and (now is None or now>=q[0].get('dispatch_not_before_s',float('-inf')))}

    def take(self, vehicle, now):
        row = self.queues[vehicle].popleft()
        if (row['model'] != row['committed_model'] or row['deadline_s'] != row['committed_deadline_s']
                or row['planned_release_s'] != row['committed_release_s']):
            raise RuntimeError('a released job commitment was changed')
        row['dispatch_s'] = now
        return row

    def rows(self):
        return [row for q in self.queues.values() for row in q]

    def cancel(self, vehicle, reason):
        for row in self.queues[vehicle]:
            row['status'] = reason
        self.queues[vehicle].clear()

    def __bool__(self):
        return any(self.queues.values())


def release_completed_io(pending,row,now):
    """Only a completed/received I/O releases the serial per-vehicle slot."""
    v=row['vehicle']
    if 'io_response_received_s' not in row:raise ValueError('I/O has not completed')
    if pending.get(v,{}).get('sample_id')!=row['sample_id']:
        raise RuntimeError('completed I/O does not own the current slot')
    pending.pop(v)
    row['pipeline_released_s']=now


def acknowledged_sources(pending,feedback):
    sources={v:record[1] for v,record in feedback.items()}
    for v,row in pending.items():
        if row.get('status')=='applied' and row.get('rpc_return_s',-1.)>sources.get(v,{}).get('rpc_return_s',-1.):
            sources[v]=row
    return sources


def periodic_successor(previous, sequence):
    period = previous['requested_period_s']
    release = previous['planned_release_s'] + period
    return dict(vehicle=previous['vehicle'], sample_id=f"{previous['vehicle']}:{sequence}",
        model=previous['model'], priority=previous['priority'], planned_release_s=release,
        minimum_release_s=release, deadline_s=release+period, nominal_deadline_s=release+period,
        requested_period_s=period, slot=previous['slot']+1, skipped_releases=0,
        event_inserted=False, status='planned', serial_vehicle=True)


def calendar_successor(previous, sequence, *, dynamic=False):
    """Keep releasing while feedback is outstanding; never renew a physical B."""
    row=periodic_successor(previous,sequence)
    row['previous_release_s']=previous['planned_release_s']
    if dynamic:
        row['deadline_s']=previous['deadline_s']
        row['feedback_pending']=True
        for field in ('obligation_actuation_s','obligation_sample_s'):
            if field in previous:row[field]=previous[field]
    return row


def replan_unreleased(candidate, last_release, sequence, now, reserve=0.):
    """Feedback can change the next arrival, never a job already published."""
    row=dict(candidate)
    target=max(last_release+.020,min(last_release+row['requested_period_s'],row['deadline_s']-reserve))
    row.update(sample_id=f"{row['vehicle']}:{sequence}",previous_release_s=last_release,
        minimum_release_s=last_release+.020,target_release_s=target,decision_available_s=now,
        planning_lateness_s=max(0.,now-target),planned_release_s=max(now,target),
        slot=sequence,feedback_pending=False)
    row['nominal_deadline_s']=row['planned_release_s']+row['requested_period_s']
    row['selected_release_interval_s']=row['planned_release_s']-last_release
    return row


def retained_unreleased(candidate, previous, last_release, sequence, now, reserve=0.):
    # Retain c/T, but use the NEW held command's physical obligation for both
    # admission alternatives. No released job is changed by this fallback.
    return replan_unreleased({**candidate,'model':previous['model'],
        'requested_period_s':previous['requested_period_s']},last_release,sequence,now,reserve)


def post_actuation_observation(candidates, actuation):
    fresh=[packet for packet in candidates if packet[2]>=actuation]
    return max(fresh,key=lambda packet:packet[0].timestamp_ns) if fresh else None


def commit_dynamic_release(row, previous, now):
    """Self-triggered planning cannot commit a release in the past."""
    target=row['planned_release_s']
    row.update(target_release_s=target, decision_available_s=now,
        planning_lateness_s=max(0.,now-target), previous_release_s=previous['planned_release_s'],
        planned_release_s=max(now,target))
    row['selected_release_interval_s']=row['planned_release_s']-row['previous_release_s']
    return row


def retained_successor(candidate,previous,now):
    """Retain configuration, not an obsolete deadline or a backdated release."""
    row={**candidate,'model':previous['model'],'requested_period_s':previous['requested_period_s']}
    row['planned_release_s']=max(now,row['minimum_release_s'],
        previous['planned_release_s']+previous['requested_period_s'])
    row['nominal_deadline_s']=row['planned_release_s']+row['requested_period_s']
    return row


def taped_release(tapes,vehicle,index,origin):
    """Diagnostic only: replay committed arrivals, not old control outputs."""
    if index>=len(tapes[vehicle]):return None
    row=dict(tapes[vehicle][index])
    if row['vehicle']!=vehicle:raise ValueError('release tape vehicle differs')
    for key in ('planned_release_s','minimum_release_s','deadline_s','nominal_deadline_s',
                'dispatch_not_before_s','delivery_not_before_s'):
        if key in row:row[key]+=origin
    row['status']='planned'
    return row
