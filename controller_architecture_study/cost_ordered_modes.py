"""A bounded minimum-cost qualification scan, without size monotonicity.

The qualification callback must include tracking/recovery AND delay validity.
This selector does not turn empirical recovery evidence into a safety proof.
"""

import math
import time


def select_mode(costs, qualify, *, max_checks, elapsed_budget_s, fallback):
    if (not costs or fallback not in costs or max_checks < 1 or elapsed_budget_s <= 0.
            or not math.isfinite(elapsed_budget_s)
            or any(not math.isfinite(c) or c <= 0. for c in costs.values())):
        raise ValueError('finite positive costs, bounded scan and an explicit fallback required')
    start = time.perf_counter()
    checked, unknown = [], False
    selected = None
    for name in sorted(costs, key=lambda m: (costs[m], m)):
        if len(checked) >= max_checks or time.perf_counter()-start >= elapsed_budget_s:
            break
        evidence = dict(qualify(name))
        if evidence.get('status') not in ('qualified', 'unqualified', 'unknown'):
            raise ValueError('qualification must distinguish unknown from rejection')
        checked.append(dict(model=name, **evidence))
        unknown |= evidence['status'] == 'unknown'
        if evidence['status'] == 'qualified':
            selected = name
            break
    elapsed = time.perf_counter()-start
    if elapsed > elapsed_budget_s:
        # A Python callback cannot be interrupted halfway through. Charge its
        # overrun and discard its result rather than pretending the cap held.
        selected = None
    return dict(model=selected or fallback, qualified=selected is not None,
                minimum_cost_proven=selected is not None and not unknown,
                empirical_only=True, fallback_used=selected is None, checked=checked,
                elapsed_s=elapsed, budget_overrun=elapsed > elapsed_budget_s,
                scope='minimum among fully qualified contracts in the supplied finite pool')


def delayed_recovery_qualification(evidence, *, delay_s, update_s, input_age_s):
    """Reuse explicit offline recovery contracts, not MSE as a safety label."""
    values = (delay_s, update_s, input_age_s)
    if not all(math.isfinite(x) and x >= 0. for x in values):
        raise ValueError('finite nonnegative timing requirements required')
    if not evidence or evidence.get('status') == 'unknown':
        return dict(status='unknown', reason='no recovery evidence for this state/domain')
    if evidence.get('status') != 'empirical_admissible':
        return dict(status='unqualified', reason='offline recovery requirement was not met')
    keys = ('delay_limit_s', 'future_update_limit_s', 'input_age_limit_s')
    if any(k not in evidence or not math.isfinite(evidence[k]) or evidence[k] < 0. for k in keys):
        return dict(status='unknown', reason='recovery evidence lacks an explicit delay/update/age contract')
    fit = all(value <= evidence[k] for value, k in zip(values, keys))
    return dict(status='qualified' if fit else 'unqualified',
                reason='empirical delayed recovery contract fits' if fit else 'timing exceeds recovery evidence',
                empirical_only=True)
