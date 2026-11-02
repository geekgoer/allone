"""Find whether the four original1.4m/s perturbations are wait-sensitive at all."""

from pathlib import Path
from unittest.mock import patch
import json

import numpy as np

from . import probe_controller_step_recoverability as stepped
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import COLLISION_CONTRACT, ControllerPool
from .framework_runtime_world import FrameworkWorld
from .probe_mode_teacher import Teacher
from .state_mode_calibration import moderate_case


SOURCE = ROOT/"controller_architecture_study/results/framework_runtime_v27_stable_static/n7_net2"
OUTPUT = ROOT/"controller_architecture_study/results/initial_recovery_window_v1"


def run():
    setup,discovery,selection = (read_json(SOURCE/f"{n}.json") for n in ("setup","discovery_summary","selection"))
    if not discovery["complete"] or setup["collision_observation"] != COLLISION_CONTRACT:
        raise ValueError("complete direct-collision comparison required")
    cases = [p["scenario"] for p in discovery["pairs"] if abs(p["scenario"]["velocity_y_mps"])==1.4]
    pool,teacher = ControllerPool(),Teacher()
    if pool.provenance() != setup["provenance"]:
        raise ValueError("controller identity differs")
    teacher_plan = ROOT/"controller_architecture_study/results/mode_teacher_probe_v2/plan.json"
    if json.loads(json.dumps(teacher.teacher.signature())) != read_json(teacher_plan)["teacher"]:
        raise ValueError("teacher identity differs")
    _,layout = moderate_case()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"source_setup_sha256":sha256(SOURCE/"setup.json"),
        "source_discovery_sha256":sha256(SOURCE/"discovery_summary.json"),"teacher_plan_sha256":sha256(teacher_plan),
        "cases":cases,"provenance":pool.provenance(),"teacher":teacher.teacher.signature(),
        "period_s":.040,"initial_waits_s":[0.,.040,.080],"duration_s":3.,
        "placement":"same reference-relative state and held control; translate to a single isolated lane",
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"mode_map_changed":False,
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_initial_recovery_window.py","probe_controller_step_recoverability.py",
             "probe_mode_teacher.py","framework_runtime.py","framework_runtime_world.py")},"test_read":False})
    rows = []
    for ci,case in enumerate(cases):
        vehicle = f"Drone{case['critical_index']+1}"
        phase = setup["phase_s"]-setup["physical_phase_span_s"]*case["critical_index"]/(setup["vehicles"]-1)
        baseline_dir = SOURCE/"discovery"/case["id"]/"static"
        first = read_json(baseline_dir/"traces.json")[vehicle][0]
        initial = read_json(baseline_dir/"initial_state.json")[vehicle]
        source_p = np.array(initial["position"])-first["reference_position"]
        source_v = np.array(initial["velocity"])-first["reference_velocity"]
        scenario = {"relative_initial":{"position_error":source_p.tolist(),"velocity_error":source_v.tolist(),
            "quaternion":initial["quaternion"],"held_control":initial["held_control"]}}
        model = selection["candidate"]["configuration"]["models"][vehicle]
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for delay in (0.,.040,.080):
                for label in (("DNN","QP_teacher") if ci%2==0 else ("QP_teacher","DNN")):
                    directory = OUTPUT/f"{case['id']}_{label}_delay{1000*delay:g}"
                    teacher.records = []
                    if label == "QP_teacher":
                        with patch.object(stepped,"encode_direct",teacher.encode):
                            report = stepped.trial(world,teacher,label,scenario,.040,directory,initial_wait=delay)
                        atomic_json(directory/"teacher_queries.json",teacher.records)
                    else:
                        report = stepped.trial(world,pool,model,scenario,.040,directory,initial_wait=delay)
                    measured = read_json(directory/"traces.json")["Drone1"][0]
                    placed = read_json(directory/"initial_state.json")["Drone1"]
                    dp = (np.array(measured["position"])-measured["reference_position"])-source_p
                    dv = (np.array(measured["velocity"])-measured["reference_velocity"])-source_v
                    matched = bool(np.max(abs(dp))<2e-5 and np.max(abs(dv))<2e-5 and all(
                        placed[k]==initial[k] for k in ("quaternion","held_control")))
                    report.update(source_case=case["id"],controller=label,reference_relative_initial_match=matched,
                        initial_relative_position_delta=dp.tolist(),initial_relative_velocity_delta=dv.tolist())
                    atomic_json(directory/"report.json",report)
                    rows.append(report)
                    atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==24,
                        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"mode_map_changed":False})
                    print(f"RECOVERY_WINDOW {len(rows)}/24 {case['id']} {label} delay={1000*delay:g} recovery={report['control_recovery_passed']} matched={matched} error={report['error']}",flush=True)
                    if report["error"] or not matched:
                        raise RuntimeError(report["error"] or "reference-relative initial state mismatch")
        finally:
            atomic_json(OUTPUT/f"cleanup_{case['id']}.json",world.close())


def main():
    placement = CpuPlacement()
    try:
        with placement:
            run()
    finally:
        if OUTPUT.exists():
            atomic_json(OUTPUT/"cpu_placement.json",placement.report)


if __name__ == "__main__":
    main()
