import numpy as np

import json
import pytest

from controller_architecture_study.probe_recovery_boundary import next_hardness,boundary_case,selected_reports,next_attempt,retain_identity
from controller_architecture_study.repair_controller_effect import route_folds
from controller_architecture_study.framework_campaign import layout_for


def test_refinement_targets_observed_transition_without_assuming_monotonicity():
    assert next_hardness({0.:(True,True),1.6:(False,False)})==.8
    # A nonmonotone response does not discard an observed transition interval.
    assert next_hardness({0.:(False,False),.8:(True,True),1.6:(False,False)})==.4


def test_identical_results_still_generate_and_record_new_queries():
    assert next_hardness({0.:(False,False),1.6:(False,False)})==.8


def test_adjacent_states_from_one_route_cannot_leak_across_cross_validation_folds():
    rows=[dict(group=f'route{route}_a{a}',cv_group=f'route{route}') for route in range(8) for a in range(5)]
    folds=route_folds(rows)
    assert set.union(*folds)=={f'route{r}' for r in range(8)}
    for route in range(8):assert sum(f'route{route}' in f for f in folds)==1


def test_boundary_stress_does_not_change_geometry_reference_or_held_control():
    base=dict(id='base',split='discovery',route_seed=3400,phase_s=2.3,layout=layout_for(0,3,3400,'continuous'),
        relative_initial=dict(position_error=[0.,0.,0.],velocity_error=[0.,0.,0.],
            quaternion=[1.,0.,0.,0.],held_control=[9.8,.1,.2,.3]))
    altered=boundary_case(base,1.6)
    assert altered['layout']==base['layout'] and altered['phase_s']==base['phase_s']
    assert altered['relative_initial']['held_control']==base['relative_initial']['held_control']
    assert base['relative_initial']['position_error']==[0.,0.,0.]
    assert abs(np.linalg.norm(altered['relative_initial']['position_error'])-.4)<1e-12
    assert abs(np.linalg.norm(altered['relative_initial']['velocity_error'])-1.92)<1e-12


def test_retry_preserves_invalid_attempt_and_reuses_valid_negative_label(tmp_path):
    directory=tmp_path/'trials'/'action';directory.mkdir(parents=True)
    failed=directory/'report.json';failed.write_text(json.dumps(dict(label=dict(valid=False))))
    attempt=next_attempt(directory);attempt.mkdir()
    valid=attempt/'report.json';valid.write_text(json.dumps(dict(label=dict(valid=True,qualified=False))))
    assert selected_reports(tmp_path)==([valid],[failed])
    assert json.loads(failed.read_text())['label']['valid'] is False


def test_identity_revision_allows_retry_code_but_never_changed_protocol(tmp_path):
    original=dict(collector_sha256='old',protocol=dict(cadence_s=.02))
    retain_identity(tmp_path,original)
    retain_identity(tmp_path,{**original,'collector_sha256':'new'})
    assert (tmp_path/'collector_revisions/new.json').exists()
    assert json.loads((tmp_path/'plan.json').read_text())==original
    with pytest.raises(RuntimeError,match='physical protocol'):
        retain_identity(tmp_path,{**original,'protocol':dict(cadence_s=.04)})
