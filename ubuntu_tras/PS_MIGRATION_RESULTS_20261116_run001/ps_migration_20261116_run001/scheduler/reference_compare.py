#!/usr/bin/env python3
"""Offline-only original-Python/C++ differential check, no board/AirSim calls.

Python 3.6 compatibility: in memory ONLY, one PEP604 type annotation in the
reference bottleneck_joint module is changed from `tuple | None` to `tuple`.
All original files and executable scheduling statements are unmodified.
"""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def compare(a, b, path='root'):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError('{} key difference {} != {}'.format(path, sorted(a), sorted(b)))
        for key in a:
            left, right = a[key], b[key]
            # Python constructs overlap by iterating an unordered vehicle set.
            # Its set membership is semantic; array order is not.
            if key == 'overlap':
                left, right = sorted(left), sorted(right)
            compare(left, right, path + '.' + key)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError('{} array size difference'.format(path))
        for i, (left, right) in enumerate(zip(a, b)):
            compare(left, right, '{}[{}]'.format(path, i))
    elif isinstance(a, float):
        if not isinstance(b, (float, int)) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10):
            raise AssertionError('{} {} != {}'.format(path, a, b))
    elif a != b:
        raise AssertionError('{} {!r} != {!r}'.format(path, a, b))


def load_reference(project):
    package = types.ModuleType('controller_architecture_study')
    package.__path__ = [str(project / 'controller_architecture_study')]
    sys.modules[package.__name__] = package
    path = project / 'controller_architecture_study/bottleneck_joint.py'
    original = path.read_text(encoding='utf-8')
    before = 'remaining_operations: tuple | None = None'
    after = 'remaining_operations: tuple = None'
    if original.count(before) != 1:
        raise ValueError('Expected exactly one known annotation; refuse unknown source shim')
    modified = original.replace(before, after)
    module = types.ModuleType('controller_architecture_study.bottleneck_joint')
    module.__file__ = str(path)
    module.__package__ = 'controller_architecture_study'
    sys.modules[module.__name__] = module
    exec(compile(modified, str(path), 'exec'), module.__dict__)
    from controller_architecture_study import obligation_scheduler as scheduler
    from controller_architecture_study.requant18_profile import ControllerMode
    hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()}
    for basename in ('obligation_scheduler.py', 'requant18_profile.py'):
        path = project / 'controller_architecture_study' / basename
        hashes[str(path.relative_to(project))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return scheduler, ControllerMode, hashes, {'before': before, 'after': after,
        'scope': 'reference interpreter in-memory type annotation only; no original file edits'}


def random_case(rng, index, mode_rows):
    ids = [row['numeric_id'] for row in mode_rows]
    jobs, permissions = [], []
    for v in range(rng.randint(1, 5)):
        vehicle = 'V{}'.format(v)
        r = rng.uniform(2., 15.)
        period = rng.uniform(6., 35.)
        allowed = rng.sample(ids, rng.randint(1, min(9, len(ids))))
        current = rng.choice(allowed)
        deadline = r + rng.uniform(.3, 18.)
        hold = r + period + rng.uniform(0., 12.)
        row = dict(name=vehicle+':0', vehicle=vehicle, release=r,
            prep_ready=r+rng.uniform(0., .8), prep_ms=rng.uniform(0., 2.),
            post_ms=rng.uniform(0., .7), transport_ms=rng.uniform(0., .8),
            slot_deadline=deadline, hold_deadline=hold,
            fresh_deadline=r+rng.uniform(2., 18.), allowed_modes=allowed,
            current_mode=current, release_options=[[r-1., deadline-1.], [r+2., deadline+2.]],
            recovery=False, mode_locked=rng.random()<.15, periodic_edit=rng.random()<.5)
        successor = dict(row)
        successor.update(name=vehicle+':1', release=r+period,
            prep_ready=row['prep_ready']+period, slot_deadline=deadline+period,
            fresh_deadline=row['fresh_deadline']+period,
            hold_deadline=hold+period, release_options=[], periodic_edit=False)
        jobs.extend((row, successor))
        permissions.append(dict(job=row['name'], successor=successor['name'],
            hold_until=hold, valid_until=r-.1, eligible=rng.random()<.6,
            evidence='synthetic differential regression'))
    rng.shuffle(jobs)
    options = dict(now=0., decision_ms=rng.choice((0., .5, 3.)),
        blocked_until=rng.choice((0., 1., 4.)), permissions=permissions,
        enable_skip=rng.random()<.7, enable_retime=rng.random()<.8,
        enable_reorder=rng.random()<.8, enable_modes=rng.random()<.9,
        max_checks=rng.randint(1,8), max_candidates=rng.randint(1,12))
    return dict(calendar=dict(jobs=jobs,revision=index,pending=[]),modes=mode_rows,options=options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--random-cases', type=int, default=240)
    args = parser.parse_args()
    project = args.bundle / 'project'
    reference, Mode, hashes, shim = load_reference(project)
    args.output.mkdir(parents=True, exist_ok=True)
    # Run the original reference scheduler unittest file under its original
    # imports; no AirSim connection or live runtime dependency is imported.
    spec = importlib.util.spec_from_file_location('test_obligation_scheduler_original',
        str(project/'tests/test_obligation_scheduler.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream_path = args.output/'reference_unittest.txt'
    with stream_path.open('w') as stream:
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        unit_result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not unit_result.wasSuccessful():
        raise ValueError('Original reference unittest failed; see '+str(stream_path))
    fixture = json.loads((args.bundle/'fixtures/scheduler_parity.json').read_text())
    requests = []
    for row in fixture['cases']:
        requests.append(dict(calendar=row['calendar'],modes=fixture['modes'],
            options=dict(now=0.,decision_ms=fixture['decision_ms'],
                permissions=row['permissions'],enable_skip=True)))
    rng=random.Random(237116)
    requests.extend(random_case(rng,i,fixture['modes']) for i in range(args.random_cases))
    rows=[]
    request_path=args.output/'active_request.json'
    for index, request in enumerate(requests):
        value=request['calendar']
        calendar=reference.Calendar(tuple(reference.ControlJob(**j) for j in value['jobs']),
            value['revision'],tuple(tuple(p) for p in value['pending']))
        modes={m['numeric_id']:Mode(**m) for m in request['modes']}
        options=dict(request['options'])
        shim_permissions=options.pop('permissions')
        options['permissions']=tuple(reference.SkipPermission(**p) for p in shim_permissions)
        expected=jsonable(reference.plan(calendar,modes,**options))
        request_path.write_text(json.dumps(request),encoding='utf-8')
        actual=json.loads(subprocess.check_output([str(args.binary),'--request',str(request_path)]).decode())
        row=dict(index=index,input=request,reference=expected,cpp=actual,status='PASS')
        try:
            compare(expected,actual)
        except AssertionError as exc:
            row.update(status='FAIL',difference=str(exc))
            rows.append(row)
            (args.output/'failed_case.json').write_text(json.dumps(row,indent=2))
            raise
        rows.append(row)
    report=dict(status='PASS',reference_unit_tests=unit_result.testsRun,
        frozen_cases=len(fixture['cases']),random_cases=args.random_cases,
        differential_cases=len(rows),all_decision_and_forecast_fields=True,
        float_abs_tolerance=1e-10,float_rel_tolerance=1e-12,
        unordered_semantic_fields=['overlap'],reference_sha256=hashes,
        compatibility_shim=shim,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        scope='offline synthetic numerical parity; no physical qualification, board or flight claim')
    (args.output/'differential_cases.json').write_text(json.dumps(rows,indent=2))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print('SCHEDULER_DIFFERENTIAL_PASS original_unit_tests={} cases={}'.format(
        unit_result.testsRun,len(rows)))


if __name__ == '__main__':
    main()
