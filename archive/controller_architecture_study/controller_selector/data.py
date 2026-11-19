"""Paired recovery protocol, signed labels and leakage-free scene groups."""

import math
import re

import numpy as np


VERSION = 'paired_recovery_selector_237d_v1'
PROTOCOL = dict(duration_s=2., observation_s=.010, cadence_s=.050, update_bound_s=.060,
    recurring_input_age_s=.010, input_age_bound_s=.020, initial_delay_tolerance_s=.010,
    position_tolerance_m=.30, velocity_tolerance_mps=.60, sustain_s=.25,
    minimum_progress_ratio=.80, net_width_m=1.7, margin_scale_m=.30,
    command_lease_s=1., maximum_observation_gap_s=.025,
    semantics='stepped physics with imposed delays; no wall-clock response or interval-wide safety certificate')


def architecture(name):
    match = re.fullmatch(r'(plain|residual)_mlp_h(\d+)_w(\d+)_seed\d+', name)
    if match is None:
        raise ValueError('an explicit supported model descriptor is required')
    kind, depth, width = match.groups()
    return np.array([int(depth)/33., int(width)/1024., float(kind == 'residual'), 2., 2.], dtype=np.float32)


def recovery_label(trace, commands, *, error_kind=None, protocol=PROTOCOL):
    """Collision is negative evidence; timing/setup errors are missing labels."""
    invalid = error_kind in ('setup', 'timing', 'infrastructure', 'interrupted')
    if invalid or not trace:
        return dict(valid=False, margin=None, qualified=None, reason=error_kind or 'missing_trace')
    collision = any(r['collision'] for r in trace)
    failed = collision or error_kind == 'flight'
    end = trace[-1]['t']
    covered = end >= protocol['duration_s']-1e-8
    tail_start = protocol['duration_s']-protocol['sustain_s']
    times = np.array([r['t'] for r in trace])
    first = max(0, int(np.searchsorted(times, tail_start, side='right'))-1)
    tail = trace[first:]
    tail_covered = bool(tail and tail[0]['t'] <= tail_start and covered)
    gap = float(np.max(np.diff(times))) if len(times) > 1 else 0.
    if (not failed and (not covered or not tail_covered)) or gap > protocol['maximum_observation_gap_s']+1e-8:
        return dict(valid=False, margin=None, qualified=None, reason='incomplete_observation_window')
    dref = np.array(trace[-1]['reference_position'])-trace[0]['reference_position']
    denom = float(dref@dref)
    if denom < .01:
        return dict(valid=False, margin=None, qualified=None, reason='insufficient_reference_progress')
    progress = float((np.array(trace[-1]['position'])-trace[0]['position'])@dref/denom)
    components = dict(safety=min(r['box_margin_m'] for r in trace)/protocol['margin_scale_m'],
        position=(protocol['position_tolerance_m']-max(r['position_error_m'] for r in tail))/protocol['position_tolerance_m'],
        velocity=(protocol['velocity_tolerance_mps']-max(r['velocity_error_mps'] for r in tail))/protocol['velocity_tolerance_mps'],
        progress=(progress-protocol['minimum_progress_ratio'])/protocol['minimum_progress_ratio'])
    margin = min(components.values())
    if failed:
        margin = min(-1., margin)
    return dict(valid=True, margin=float(margin), qualified=bool(margin >= 0.),
        components=components, collision=collision, tail_covered=tail_covered,
        progress_ratio=progress, observed_duration_s=end, reason='observed_failure' if failed else 'window_complete')


def paired_groups(rows, models, delays):
    """Retain a scene only when every candidate and delay has a valid label."""
    grouped = {}
    expected = {(m, float(d)) for m in models for d in delays}
    for row in rows:
        grouped.setdefault(row['group'], []).append(row)
    valid, rejected = [], {}
    for key, group in grouped.items():
        observed = {(r['model'], float(r['requested_delay_s'])) for r in group}
        if len(group) != len(expected) or observed != expected or not all(r['label']['valid'] for r in group):
            rejected[key] = 'incomplete_or_invalid_pair_block'
            continue
        first = np.asarray(group[0]['state_feature'], dtype=float)
        if any(not np.allclose(first, r['state_feature'], rtol=0., atol=2e-4) for r in group):
            rejected[key] = 'initial_feature_mismatch'
            continue
        if len({r['split'] for r in group}) != 1:
            raise ValueError('one scene cannot occur in multiple training/calibration/test splits')
        valid.extend(group)
    return valid, rejected


def simultaneous_calibration(predicted, target, groups, alpha):
    if not 0. < alpha < 1.:
        raise ValueError('alpha must be between zero and one')
    residual = np.asarray(predicted)-np.asarray(target)
    values = np.array([np.max(residual[np.asarray(groups) == g]) for g in sorted(set(groups))])
    rank = math.ceil((len(values)+1)*(1.-alpha))
    if rank > len(values) or not len(values):
        return dict(correction=None, groups=len(values), rank=rank, available=False,
                    reason='not enough independent calibration scenes for this alpha')
    return dict(correction=float(max(0., np.sort(values)[rank-1])), groups=len(values), rank=rank,
                available=True, scope='simultaneous finite recorded candidate/delay grid under exchangeability; not closed-loop safety')
