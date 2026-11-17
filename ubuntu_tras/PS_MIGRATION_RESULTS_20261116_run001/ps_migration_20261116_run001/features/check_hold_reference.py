#!/usr/bin/env python3
"""Offline parity only: execute frozen numeric functions with annotation shim.

The source handoff is never modified. Only type annotations of selected helper
functions are removed in memory, to permit reading modern reference code in the
host's Python 3.6. No numerical statement is replaced or patched. This script is
NOT part of the board control runtime.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


class StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return self.generic_visit(node)


def numeric_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(found) != len(names):
        raise ValueError("Reference helper name mismatch: " + str(path))
    module = ast.Module(body=found)
    module = ast.fix_missing_locations(StripAnnotations().visit(module))
    exec(compile(module, str(path), "exec"), namespace)


def load_reference(project):
    # Imported helpers retain the exact arithmetic/branching in their bodies.
    state_path = project / "state_adapter.py"
    state_namespace = {"np": np}
    numeric_functions(state_path, {"_quaternion", "normalize_quaternion", "quaternion_multiply"}, state_namespace)
    route_namespace = dict(state_namespace)
    numeric_functions(project / "gate_route.py", {"_rotate_vector"}, route_namespace)
    route = ModuleType("gate_route")
    route._rotate_vector = route_namespace["_rotate_vector"]
    sys.modules["gate_route"] = route
    visual_namespace = {"np": np, "_rotate_vector": route._rotate_vector}
    numeric_functions(project / "gate_visualization.py", {"_normalise_quaternion", "_quaternion_multiply", "_rotate_vector", "_gate_parts"}, visual_namespace)
    visual = ModuleType("gate_visualization")
    visual._gate_parts = visual_namespace["_gate_parts"]
    sys.modules["gate_visualization"] = visual
    path = project / "controller_architecture_study/held_control_safety.py"
    spec = importlib.util.spec_from_file_location("frozen_held_control_safety", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--cpp-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = args.handoff / "project"
    reference = load_reference(project)
    scene = json.loads((project / "controller_architecture_study/results/release001_live_case_20261116/case.json").read_text())
    cpp = json.loads(args.cpp_report.read_text())
    bounds = reference.HoldBounds(2., .01, .05, .15, 10.,
        "Route-B development assumptions; not plant-certified")
    analyzers = {}
    for vehicle, layout in scene["layouts"].items():
        gates = [SimpleNamespace(index=g["index"], center=np.array(g["center"]),
                  orientation=np.array(g["orientation"])) for g in layout["gates"]]
        analyzers[vehicle] = reference.GateHoldAnalyzer(gates,
            scene["gate_center_width_m"]-.15, bounds, net_height=4.-.15)
    results = []
    maximum = 0.
    for i, row in enumerate(cpp["hold_cases"]):
        state = SimpleNamespace(**{key: np.asarray(value) for key, value in row["state"].items()
            if key != "timestamp_ns"})
        control = np.array(row["effective_control"])
        analyzer = analyzers[row["vehicle"]]
        expected = analyzer.analyze(state, control, details=True)
        batch = reference.GateHoldBatch({row["vehicle"]: analyzer})
        batched = batch.analyze({row["vehicle"]: state}, {row["vehicle"]: control})[row["vehicle"]]
        expected["mesh_margin_m"] = batched["mesh_margin_m"]
        expected["mesh_rate_mps"] = batched["mesh_rate_mps"]
        actual = row["actual"]
        errors = {}
        for key in ("budget_s", "nominal_acceleration", "per_obstacle_period_s", "mesh_margin_m", "mesh_rate_mps"):
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-9, rtol=2e-9,
                err_msg="case {} {}".format(i, key))
            errors[key] = float(np.max(np.abs(np.asarray(actual[key])-np.asarray(expected[key]))))
            maximum = max(maximum, errors[key])
        for key in ("valid", "binding_obstacle", "separating_face", "plant_bounds_verified", "is_safety_certificate"):
            if actual[key] != expected[key]:
                raise ValueError("case {} mismatch {}".format(i, key))
        results.append(dict(case=i, vehicle=row["vehicle"], errors=errors,
            budget_s=actual["budget_s"], binding_obstacle=actual["binding_obstacle"]))
    files = ["state_adapter.py", "gate_route.py", "gate_visualization.py",
        "controller_architecture_study/held_control_safety.py"]
    report = dict(status="PASS", cases=len(results), maximum_abs_difference=maximum,
        scope="offline numeric parity only; not flight/plant safety verification",
        shim="selected helper type annotations stripped in memory, numeric bodies unchanged",
        python=sys.version, numpy=np.__version__, cases_detail=results,
        reference_sha256={name: hashlib.sha256((project/name).read_bytes()).hexdigest() for name in files})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("HOLD_REFERENCE_PARITY_PASS cases={} max_abs={:.9g}".format(len(results), maximum))


if __name__ == "__main__":
    main()
