#!/usr/bin/env python3
"""Native prepare geometry/management parity. Mock != flight/PS evidence."""
import argparse
import json
import math
import pathlib
import subprocess
import tempfile


def read(path):
    with open(str(path), encoding="utf-8") as stream:
        return json.load(stream)


def poly(layout, t, derivative=0):
    segment = layout["segments"][-1]
    for candidate in layout["segments"]:
        if t < candidate["start_time"] + candidate["duration"]:
            segment = candidate
            break
    dt = max(0., min(segment["duration"], t-segment["start_time"]))
    return [sum(c[k] * (math.factorial(k)/math.factorial(k-derivative)) * dt**(k-derivative)
                for k in range(derivative, 6)) for c in segment["coefficients"]]


def near(a, b, limit=1e-9):
    if len(a) != len(b) or max(abs(x-y) for x, y in zip(a, b)) > limit:
        raise AssertionError((a, b))


def audit(data, case, vehicles, offsets):
    extents = [[poly(case["layouts"][v], j*case["duration_s"]/200.) for j in range(201)]
               for v in vehicles]
    means = [sum(p[1] for p in points)/201. for points in extents]
    spacing = max(12., max(max(p[1] for p in points)-min(p[1] for p in points)
                           for points in extents)+10.)
    near([data["spacing_m"]], [spacing])
    for index, vehicle in enumerate(vehicles):
        shift = [-offsets[vehicle][0],
                 (index-(len(vehicles)-1)/2.)*spacing-means[index]-offsets[vehicle][1],
                 -offsets[vehicle][2]]
        near(data["shifts"][vehicle], shift)
        old, new = case["layouts"][vehicle], data["layouts"][vehicle]
        for before, after in zip(old["segments"], new["segments"]):
            for axis in range(3):
                near(after["coefficients"][axis],
                     [before["coefficients"][axis][0]+shift[axis]]+before["coefficients"][axis][1:])
        for before, after in zip(old["gates"], new["gates"]):
            near(after["center"], [a+b for a, b in zip(before["center"], shift)])
        initial = data["initial_states"][vehicle]
        near(initial["position"], poly(new, 1.5))
        near(initial["velocity"], poly(new, 1.5, 1))
    assert len(data["gate_objects"]) == 20*len(vehicles)
    assert data["effective_control"] == [9.81, 0., 0., 0.]
    for actor in data["gate_objects"]:
        assert actor["name"].startswith("PSMIG20261116_")
        assert not actor["physics_enabled"] and not actor["is_blueprint"]
        vehicle = actor["vehicle"]
        gate = next(g for g in data["layouts"][vehicle]["gates"] if g["index"] == actor["gate_index"])
        delta = {"left": [0, -2, 0], "right": [0, 2, 0],
                 "top": [0, 0, -2], "bottom": [0, 0, 2]}[actor["part"]]
        # Frozen case gates have identity orientation; nonidentity rotation is
        # separately covered by the numerical feature fixture suite.
        assert gate["orientation"] == [1., 0., 0., 0.]
        near(actor["position"], [gate["center"][i]+offsets[vehicle][i]+delta[i] for i in range(3)])
        multiplier = data["asset_scale_multiplier"]
        size = [.15, .15, 4.] if actor["part"] in ("left", "right") else [.15, 4., .15]
        near(actor["scale"], [v*multiplier for v in size])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="/home/cwc/dpu_work/windows_framework/route_b_ps_handoff_20261116")
    args = parser.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    bundle = pathlib.Path(args.bundle)
    case_path = bundle/"project/controller_architecture_study/results/release001_live_case_20261116/case.json"
    calibration = bundle/"project/configs/thrust_calibration.json"
    case = read(case_path)
    results = pathlib.Path(tempfile.mkdtemp(prefix="prepare_tests_", dir=str(here)))
    binary = here/"build_host/prepare_case"
    common = [str(binary), "--case", str(case_path), "--calibration", str(calibration), "--reference-start-s", "1.5"]
    for vehicles in (["Drone1"], ["Drone1", "Drone2"]):
        offsets = {"Drone1": [2., -3., 0.5], "Drone2": [-1., 20., -0.5]}
        path = results/("offsets%d.json" % len(vehicles))
        path.write_text(json.dumps(offsets), encoding="utf-8")
        output = results/("offline%d" % len(vehicles))
        command = common+["--offline", "--offsets", str(path), "--vehicles", ",".join(vehicles), "--out", str(output)]
        record = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        (results/("offline%d.log" % len(vehicles))).write_text(record.stdout, encoding="utf-8")
        assert record.returncode == 0, record.stdout
        data = read(output/"setup.json")
        assert data["status"] == "OFFLINE_LAYOUT_ONLY" and not (output/"prepared_case.json").exists()
        audit(data, case, vehicles, offsets)
    for foreign in (False, True):
        logfile = results/("mock_foreign.log" if foreign else "mock_live.log")
        with logfile.open("w", encoding="utf-8") as log:
            server = subprocess.Popen([str(here/"build_host/mock_prepare_server"), "43464"]+(["foreign"] if foreign else []),
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, universal_newlines=True)
            try:
                assert server.stdout.readline().strip() == "MOCK_PREPARE_READY"
                output = results/("live_foreign" if foreign else "live_ok")
                command = common+["--host", "127.0.0.1", "--port", "43464", "--vehicles", "Drone1,Drone2", "--out", str(output)]
                record = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                (results/("prepare_foreign.log" if foreign else "prepare_live.log")).write_text(record.stdout, encoding="utf-8")
                data = read(output/"setup.json")
                if foreign:
                    assert record.returncode != 0 and data["status"] == "FAILED"
                    assert not data["simulation_touched"]
                    assert not (output/"prepared_case.json").exists()
                else:
                    assert record.returncode == 0, record.stdout
                    assert data["status"] == "PREPARED_PAUSED" and data["physics_paused"]
                    assert len(data["created_actors"]) == 40
                    assert data["asset"] == "EditorCube" and data["asset_scale_multiplier"] == .5
                    assert data["previous_owned_actors"] == ["PSMIG20261116_previous_owned_actor"]
                    assert data["cleanup_results"] == {"PSMIG20261116_previous_owned_actor": True}
                    assert data["cleanup_remaining"] == [] and data["simulation_touched"]
                    audit(data, case, ["Drone1", "Drone2"], {"Drone1": [0,0,0], "Drone2": [0,20,0]})
                    assert (output/"prepared_case.json").is_file()
            finally:
                if server.poll() is None:
                    output, _ = server.communicate("quit\n", timeout=3)
                    (results/("server_foreign.txt" if foreign else "server_live.txt")).write_text(output, encoding="utf-8")
                    if foreign:
                        assert "RESET" not in output and "DESTROY" not in output
    offsets = results/"offsets1.json"
    valid_prefix = common+["--offline", "--offsets", str(offsets), "--vehicles", "Drone1",
                          "--prefix", "PSMIG20261116_repeat001_", "--out", str(results/"prefix_ok")]
    record = subprocess.run(valid_prefix, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    (results/"prefix_ok.log").write_text(record.stdout, encoding="utf-8")
    assert record.returncode == 0, record.stdout
    for index, prefix in enumerate(("PSMIG20261116_.*", "UserScene_", "PSMIG20261116_../")):
        output = results/("prefix_reject%d" % index)
        record = subprocess.run(common+["--offline", "--offsets", str(offsets), "--vehicles", "Drone1",
                                       "--prefix", prefix, "--out", str(output)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        (results/("prefix_reject%d.log" % index)).write_text(record.stdout, encoding="utf-8")
        assert record.returncode != 0 and not output.exists()
    print("PREPARE_CASE_PARITY_PASS offline_single,offline_dual,live_mock_dual,owned_cleanup,foreign_guard,prefix_guards RESULTS", results)


if __name__ == "__main__":
    main()
