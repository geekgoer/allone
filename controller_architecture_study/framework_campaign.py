"""Freeze the working prototype and prepare diverse, paired full-flight cases.

Preparation does not launch AirSim or claim the research policy is deployed.
"""

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from falcon_paper_fidelity.motion_primitive import QuinticPolynomial
from .decision_dataset import ROOT, atomic_json, sha256


REFERENCE = ROOT/'controller_architecture_study/results/joint_runtime_repair_20261029_v3'
FAMILIES = ('smooth', 'late', 'mixed')


def layout_for(index, count, seed, family):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, count]))
    if family == 'mixed':
        kind = ('continuous', 'smooth', 'late')[min(2, 3*index//count)]
    else:
        kind = family
    onset = int(rng.integers(6, 8)) if kind == 'late' else 1 if kind == 'continuous' else 99
    amplitude = float(rng.uniform(.40, .62) if kind != 'smooth' else rng.uniform(.02, .10))
    p, v, zero = np.array([0., 0., -4.]), np.array([5., 0., 0.]), np.zeros(3)
    segments, gates, clock = [], [], 0.
    sign = float(rng.choice([-1., 1.]))
    for k in range(11):
        dt = float(rng.uniform(1.25, 1.45)) if k not in (0, 10) else 2.1
        q = p.copy()
        q[0] += 5.*dt*(.5 if k == 10 else 1.)
        if k < 10:
            if k >= onset:
                # Independent bend severity/direction and spacing, not the old
                # seed changing only a ten-percent global amplitude factor.
                sign *= -1. if rng.random() < .8 else 1.
                q[1] = sign*amplitude*float(rng.uniform(.65, 1.))
                q[2] = -4.+float(rng.uniform(-.12, .12))
            elif kind == 'smooth':
                q[1] = amplitude*np.sin(.6*k+seed)
        end_v = zero if k == 10 else np.array([5., 0., 0.])
        polynomial = QuinticPolynomial(p, v, zero, q, end_v, zero, dt)
        segments.append(dict(start_time=clock, duration=dt,
                             coefficients=polynomial.coefficients.tolist(), heading_orientation=[1., 0., 0., 0.]))
        if k < 10:
            gates.append(dict(index=k+1, center=q.tolist(), orientation=[1., 0., 0., 0.]))
        p, v, clock = q, end_v, clock+dt
    return dict(segments=segments, gates=gates, duration=clock, family=kind,
                seed=seed, nominal_speed=5., bend_amplitude_m=amplitude, hard_onset_gate=onset+1,
                duration_fitting='frozen natural geometry; no outcome-driven slowdown')


def make_scene(count, seed, family):
    layouts = {f'Drone{i+1}': layout_for(i, count, seed, family) for i in range(count)}
    duration = np.ceil((max(x['duration'] for x in layouts.values())+1.68)/.42)*.42
    return dict(id=f'n{count}_{family}_seed{seed}', count=count, layouts=layouts,
                duration_s=float(duration), net_width_m=1.7, phase_s=0., phase_span_s=0.,
                virtual_extra_s=0., cache_interval_s=[2.1, 4.2],
                external_impulse=False, initial_tracking_offset=False,
                prefix_estimate='disabled: all scored windows must be physically flown')


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/'working_prototype'
    shutil.copytree(REFERENCE/'frozen_source', frozen)
    evidence = root/'baseline_evidence'
    evidence.mkdir()
    for name in ('plan.json', 'summary.json', 'comparison_audit.json'):
        shutil.copy2(REFERENCE/name, evidence/name)
    archive = root/'working_prototype.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(frozen.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(root))
    cases = []
    for n in (3, 6, 10):
        for family in FAMILIES:
            for seed in range(1000, 1010):
                case = make_scene(n, seed, family)
                path = root/'scenes'/(case['id']+'.json')
                atomic_json(path, case)
                cases.append(dict(id=case['id'], path=str(path.relative_to(root)), sha256=sha256(path),
                                  count=n, family=family, seed=seed))
    protocol = dict(date='2026-10-30', cases=cases, arms=['static', 'route_preplanned', 'state_feedback'],
        paired_scenes=len(cases), planned_flights=3*len(cases), full_routes_physically_flown=True,
        calibration_seeds=[101, 102], heldout_seeds=list(range(1000, 1010)),
        static_search_budget=20, static_objective='largest finite full-flight J on calibration runs, no collision filter',
        static_calibration='full nominal trajectories without physical gate meshes; all evaluation arms use physical meshes',
        static_search_exact_global_optimum=False, full_FALCON_SVM=False,
        common_runtime=['cpp_io', 'eager_completion', 'frontend_aging', 'same_model_pool',
                        'same_core_placement', 'same_queue_policy', 'single_vehicle_collision_retirement'],
        repeated_vehicle_samples_are_independent=False, evaluation_unit='complete paired multi-vehicle scene',
        stage='prepared, NOT flown; research policy integration and calibration still required',
        plant_bounds_verified=False, legacy_snapshot_sha256=sha256(archive))
    atomic_json(root/'campaign_protocol.json', protocol)
    atomic_json(root/'working_prototype_hashes.json',
                {str(p.relative_to(frozen)): sha256(p) for p in sorted(frozen.rglob('*')) if p.is_file()})
    print(json.dumps(dict(output=str(root), preserved_prototype=str(archive),
                          paired_scenes=len(cases), planned_flights=3*len(cases), flights_started=0), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)


if __name__ == '__main__':
    main()
