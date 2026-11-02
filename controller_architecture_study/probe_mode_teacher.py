"""Use the original label teacher on the exact matched context-probe states."""

from pathlib import Path
from unittest.mock import patch
import json

from . import probe_controller_step_recoverability as stepped
from .convex_teacher import ConvexTeacher
from .decision_dataset import ROOT, atomic_json, sha256
from .dynamic_probe_common import read_json
from .framework_cpu_affinity import CpuPlacement
from .framework_runtime import COLLISION_CONTRACT
from .framework_runtime_world import FrameworkWorld
from .state_mode_calibration import moderate_case


SOURCE = ROOT/"controller_architecture_study/results/mode_projection_probe_v1"
PERIOD = ROOT/"controller_architecture_study/results/mode_period_probe_v1"
CAPACITY = ROOT/"controller_architecture_study/results/mode_context_capacity_v1"
OUTPUT = ROOT/"controller_architecture_study/results/mode_teacher_probe_v2"


class Teacher:
    def __init__(self):
        self.teacher = ConvexTeacher()
        self.encoder = stepped.encode_direct
        self.records = []

    def encode(self, window, state, elapsed):
        self.problem = state,window.window(elapsed)
        return self.encoder(window,state,elapsed)

    def predict(self, name, feature):
        action,diagnostic = self.teacher.query(*self.problem)
        diagnostic["first_acceleration"] = diagnostic["first_acceleration"].tolist()
        self.records.append({"control":action.tolist(),**diagnostic})
        return action


def run():
    plan,capacity = read_json(SOURCE/"plan.json"),read_json(CAPACITY/"summary.json")
    if not capacity["complete"] or not all(r["initial_state_match"] for r in capacity["rows"]):
        raise ValueError("completed matched six-model comparison required")
    controller = Teacher()
    manifest = ROOT/"controller_architecture_study/results/decision_convex_96k_v1/manifest.json"
    original = read_json(manifest)["config"]["teacher"]
    if json.loads(json.dumps(controller.teacher.signature()["settings"])) != original["settings"]:
        raise ValueError("QP settings differ from the frozen label teacher")
    _,layout = moderate_case()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    atomic_json(OUTPUT/"plan.json",{"source_plan_sha256":sha256(SOURCE/"plan.json"),
        "capacity_summary_sha256":sha256(CAPACITY/"summary.json"),"teacher_manifest_sha256":sha256(manifest),
        "teacher":controller.teacher.signature(),"label_teacher_settings_match":True,
        "contexts":plan["phases"],"periods_s":{"captured_phase":[.040],"calibration_phase":[.040,.020]},
        "sides":[1,-1],"initial_wait_s":0.,"collision_observation":COLLISION_CONTRACT,
        "encoder_hook":"observe exact state/references; return unchanged physical237D feature",
        "source_sha256":{n:sha256(Path(__file__).with_name(n)) for n in
            ("probe_mode_teacher.py","probe_controller_step_recoverability.py","convex_teacher.py",
             "framework_runtime_world.py","framework_runtime_features.py")},
        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"test_read":False})
    rows = []
    for context,phase in plan["phases"].items():
        world = FrameworkWorld(layout,phase,1,2.)
        try:
            for side in (1,-1):
                source_dir = PERIOD/f"{context}_side{side:+d}_delay0"
                previous = read_json(source_dir/"report.json")
                initial,measured = read_json(source_dir/"initial_state.json")["Drone1"],read_json(source_dir/"traces.json")["Drone1"][0]
                for period in ((.040,) if context=="captured_phase" else (.040,.020)):
                    directory = OUTPUT/f"{context}_side{side:+d}_T{1000*period:g}"
                    controller.records = []
                    with patch.object(stepped,"encode_direct",controller.encode):
                        report = stepped.trial(world,controller,"QP_teacher",previous["scenario"],period,directory)
                    other,observed = read_json(directory/"initial_state.json")["Drone1"],read_json(directory/"traces.json")["Drone1"][0]
                    matched = all(initial[k] == other[k] for k in ("position","velocity","quaternion","held_control")) and all(
                        measured[k] == observed[k] for k in ("position","velocity","quaternion"))
                    report.update(context=context,side=side,initial_state_match=matched,
                        max_qp_constraint_violation=max((r["constraint_violation"] for r in controller.records),default=None))
                    atomic_json(directory/"teacher_queries.json",controller.records)
                    atomic_json(directory/"report.json",report)
                    rows.append(report)
                    atomic_json(OUTPUT/"summary.json",{"rows":rows,"complete":len(rows)==6,
                        "simulation_stepped":True,"end_to_end_realtime_claimed":False,"mode_map_changed":False})
                    print(f"TEACHER {len(rows)}/6 {context} side={side} T{1000*period:g} recovery={report['control_recovery_passed']} matched={matched} error={report['error']}",flush=True)
                    if report["error"] or not matched:
                        raise RuntimeError(report["error"] or "teacher initial conditions differ")
        finally:
            atomic_json(OUTPUT/f"cleanup_{context}.json",world.close())


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
