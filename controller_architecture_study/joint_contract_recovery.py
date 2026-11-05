"""Maintain UNPUBLISHED safety obligations across actual control epochs.

Release-slot IDs stay unchanged. Obligation versions and prior violations are
recorded separately. Same-input expiry needs post-expiry evidence; a new
acknowledged input ends the previous hold. No native calendar is changed.
Inputs are held-control safety obligations, not arbitrary computational-job
response deadlines. The latter retain their original miss accounting.
"""

import math


def prepare_recovery(now, future, pending, certificates, *, reserve_s=.006):
    result={v:dict(row) for v,row in future.items()}
    renewals=[];deferred=[];history=[];previous_obligations=[]
    outstanding={r['vehicle'] for r in pending if 'command_completed_s' not in r}
    # Only a dispatched command can already have changed the plant while its
    # ACK is unknown. Sampling/compute remains fully charged by the forecast;
    # it must not keep a fulfilled OLD hold alive on an unpublished successor.
    command_inflight={r['vehicle'] for r in pending if 'command_completed_s' not in r
        and r.get('stage') not in ('input','sampling','input_ready','compute','output')}
    for v,row in result.items():
        old=row['deadline_s'];cert=certificates.get(v,{})
        if (cert.get('available') and 'obligation_actuation_s' in row and
                cert.get('actuation_s',-float('inf'))<row['obligation_actuation_s']-1e-9):
            deferred.append(dict(vehicle=v,sample_id=row['sample_id'],reason='certificate_control_epoch_older_than_plan'))
            continue
        changed_control=(cert.get('available',False) and 'obligation_actuation_s' in row
            and cert.get('actuation_s',-float('inf'))>row['obligation_actuation_s']+1e-9)
        if old>now and not changed_control:
            # A nonexpired commitment may only tighten, never be postponed.
            if cert.get('available'):row['deadline_s']=min(old,cert['deadline_s'])
            continue
        fulfilled=changed_control and cert['actuation_s']<=old
        previous_obligations.append(dict(vehicle=v,sample_id=row['sample_id'],deadline_s=old,
            status='ended_by_new_command_ack_before_deadline' if fulfilled else 'deadline_already_missed',
            fulfillment_is_rpc_proxy=True))
        if not fulfilled:
            history.append(dict(vehicle=v,sample_id=row['sample_id'],deadline_s=old,
                                status='deadline_already_missed_before_recovery'))
        reason=None
        # A new ACK is a fact about the held control, not permission to move a
        # release. Rebind an unpublished successor even inside the configuration
        # planning fence; the caller still freezes its release and uses native
        # publication/CAS guards. Already due/published work remains immutable.
        fence=now if changed_control else now+reserve_s
        if row.get('stage','input')!='input' or row['planned_release_s']<=fence:
            reason='publication_or_planning_fence'
        elif v in command_inflight or (v in outstanding and not changed_control):
            reason='unacknowledged_control_work_may_change_held_input'
        elif not cert.get('available'):
            reason='no_paired_state_and_actual_control'
        elif not math.isfinite(cert.get('sample_s',float('nan'))) or cert['sample_s']>now:
            reason='observation_clock_not_current'
        elif cert['sample_s']<cert.get('actuation_s',-float('inf')):
            reason='observation_before_current_control_ack'
        elif not changed_control and cert['sample_s']<=old:
            reason='observation_not_after_expired_obligation'
        elif cert['deadline_s']<=fence:
            reason='fresh_certificate_has_no_remaining_planning_time'
        if reason:
            deferred.append(dict(vehicle=v,sample_id=row['sample_id'],reason=reason))
            continue
        version=f"{row['sample_id']}@ack:{cert.get('actuation_s',0.):.9f}@observation:{cert['sample_s']:.9f}"
        row.update(deadline_s=cert['deadline_s'],obligation_version=version,
                   superseded_deadline_s=old,obligation_sample_s=cert['sample_s'],
                   deadline_source='new_held_control_epoch' if changed_control else 'fresh_post_expiry_held_control_certificate')
        if 'actuation_s' in cert:row['obligation_actuation_s']=cert['actuation_s']
        renewals.append(dict(vehicle=v,sample_id=row['sample_id'],previous_deadline_s=old,
            new_deadline_s=row['deadline_s'],observation_s=cert['sample_s'],obligation_version=version,
            new_control_epoch=changed_control,previous_obligation_fulfilled=fulfilled,
            metadata_only_inside_planning_fence=changed_control and row['planned_release_s']<=now+reserve_s,
            outstanding_work_still_charged=[r['sample_id'] for r in pending if r['vehicle']==v],
            conditional_only=not cert.get('is_safety_certificate',False)))
    return result,dict(renewals=renewals,deferred=deferred,previous_misses=history,previous_obligations=previous_obligations,
        historical_deadlines_repaired=False,installed=False,
        scope='new future obligation; no claim of continuous safety through the previous gap')
