"""Switch scheduling, immutable D=T and mocked flight; no real GPU/AirSim."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from controller_architecture_study import dynamic_probe_common as common
from controller_architecture_study import probe_airsim_three as live
from controller_architecture_study.joint_switch_flight import SwitchPlan
from controller_architecture_study.audit_joint_switch import legacy_slots, recompute


def plan():
    steps = [dict(at_seconds=t, model=m, period_ms=p) for t, m, p in
             [(0., "plain4x256", 50.), (.1, "plain4x512", 75.),
              (.25, "plain4x512", 100.), (.45, "plain4x256", 50.)]]
    return SwitchPlan(steps, steps)


class ScheduleTests(unittest.TestCase):
    def test_slot_counts_do_not_depend_on_host_clock_origin(self):
        steps = [dict(at_seconds=t, model=m, period_ms=p) for t, m, p in
                 [(0., "plain4x256", 50.), (3., "plain4x512", 75.),
                  (6., "plain4x512", 100.), (9., "plain4x256", 50.)]]
        value = SwitchPlan(steps, steps)
        for phase in (0., .05/3, .1/3):
            baseline = list(value.slots(0., 20., phase))
            self.assertEqual(len(baseline), 350)
            for start in (263113.7462448, 1e9):
                shifted = list(value.slots(start, 20., phase))
                self.assertEqual(len(shifted), len(baseline))
                self.assertEqual([c for _, _, c in shifted], [c for _, _, c in baseline])
                np.testing.assert_allclose([s-start for _, s, _ in shifted],
                    [s for _, s, _ in baseline], rtol=0., atol=2e-7)

    def test_switch_preserves_committed_next_release_and_clock(self):
        slots = list(plan().slots(10., .6))
        np.testing.assert_allclose([s for _, s, _ in slots],
            [10., 10.05, 10.1, 10.175, 10.25, 10.35, 10.45, 10.5, 10.55])
        for (_, before, choice), (_, after, _) in zip(slots, slots[1:]):
            self.assertAlmostEqual(after-before, choice["period_ms"]/1000)
            self.assertEqual(choice["period_ms"], choice["deadline_ms"])
        shifted = list(plan().slots(10., .6, 50/3000))
        self.assertAlmostEqual(shifted[0][1], 10.+50/3000)

    def test_old_deadline_and_plan_are_immutable(self):
        value = plan()
        first = value.at(0.)
        value.at(.3)["deadline_ms"] = 1000.
        self.assertEqual(first["deadline_ms"], 50.)
        self.assertEqual(value.at(.3)["deadline_ms"], 100.)
        rows = [dict(vehicle="a", status="ready", scheduled_s=0., sample_start_s=0.,
                     enqueue_s=0., start_s=0., ready_s=.075, service_ms=75.,
                     dnn_dispatch_s=.076, deadline_ms=t) for t in (50., 100.)]
        result = common.summarize_timing(rows, None)
        self.assertEqual(result["output_deadline_misses"], 1)
        self.assertEqual(result["dnn_dispatch_deadline_misses"], 1)

    def test_invalid_or_unflown_pairs_are_rejected(self):
        steps = [dict(at_seconds=0., model="small", period_ms=50.)]
        for bad in ([], [{**steps[0], "at_seconds": 1.}],
                    [steps[0], steps[0]], [{**steps[0], "period_ms": 70.}],
                    [{**steps[0], "at_seconds": float("nan")}],
                    [{**steps[0], "model": "untrained"}]):
            with self.assertRaises(ValueError):
                SwitchPlan(bad, steps)

    def test_skipped_slots_keep_their_configuration(self):
        class Service:
            def __init__(self): self.rows = []
            def record_skip(self, vehicle, cycle, scheduled, **metadata):
                self.rows.append(dict(vehicle=vehicle, cycle=cycle, scheduled_s=scheduled, **metadata))
        service = Service()
        with patch("controller_architecture_study.joint_switch_flight.time.perf_counter", return_value=1.):
            self.assertEqual(list(plan().releases(0., .6, 0., threading.Event(), service, "a")), [])
        self.assertGreater(len(service.rows), 4)
        self.assertTrue(all(r["deadline_ms"] == r["period_ms"] for r in service.rows))


class AuditTests(unittest.TestCase):
    def fixture(self):
        steps = [dict(at_seconds=t, model=m, period_ms=p) for t, m, p in
                 [(0., "plain4x256", 50.), (3., "plain4x512", 75.),
                  (6., "plain4x512", 100.), (9., "plain4x256", 50.)]]
        value, rows = SwitchPlan(steps, steps), []
        for i, vehicle in enumerate(live.VEHICLES):
            for cycle, scheduled, choice in legacy_slots(value, 263113.7462448, 20., i*.05/3):
                rows.append(dict(vehicle=vehicle, cycle=cycle, scheduled_s=scheduled, **choice,
                    status="ready", sample_start_s=scheduled, enqueue_s=scheduled,
                    start_s=scheduled, ready_s=scheduled+.001, service_ms=1.,
                    dnn_dispatch_s=scheduled+.002, command_duration_ms=choice["period_ms"]))
        source = dict(experiment="scripted_joint_switch_mechanism", phase="staggered",
            duration_seconds=20., switch_plan=steps, status="COMPLETE", errors=[],
            planned_releases=1050, flight_passed=True, realtime_passed=False,
            acceptance_passed=False, clock_ratio_passed=True, tracking_window_covered=True,
            tracking_rmse_m=.36, limitations=[])
        return source, rows

    def test_legacy_endpoint_is_audited_without_rewriting_original_flags(self):
        source, rows = self.fixture()
        result = recompute(source, rows)
        self.assertEqual(result["legacy_host_schedule_releases"], 1051)
        self.assertEqual(result["repaired_relative_schedule_releases"], 1050)
        self.assertTrue(result["corrected_acceptance_passed"])
        self.assertFalse(source["acceptance_passed"])
        self.assertIsNotNone(result["command_input_age"]["mean_age_when_served_ms"])
        self.assertLess(result["command_input_age"]["unserved_fraction"], .01)

    def test_missing_duplicate_or_wrong_configuration_still_fails(self):
        source, rows = self.fixture()
        for invalid in (rows[:-1], rows+[dict(rows[-1])],
                        [{**rows[0], "model": "wrong"}]+rows[1:]):
            self.assertFalse(recompute(source, invalid)["corrected_acceptance_passed"])

    def test_real_deadline_miss_still_fails(self):
        source, rows = self.fixture()
        rows[0]["dnn_dispatch_s"] = rows[0]["scheduled_s"]+.06
        result = recompute(source, rows)
        self.assertEqual(result["timing"]["dnn_dispatch_deadline_misses"], 1)
        self.assertFalse(result["corrected_acceptance_passed"])

    def test_short_or_failed_flight_is_not_promoted_to_pass(self):
        source, rows = self.fixture()
        for key in ("flight_passed", "clock_ratio_passed", "tracking_window_covered"):
            self.assertFalse(recompute({**source, key: False}, rows)["corrected_acceptance_passed"])


class FlightTests(unittest.TestCase):
    def test_hold_period_rpc_ownership_and_single_placement(self):
        resets, placements, commands = [], [], []
        first = SimpleNamespace(position=np.array([0., 0., -4.]), orientation=np.array([1., 0., 0., 0.]))
        class Client:
            def __init__(self, **kwargs): self.owner = threading.get_ident()
            def confirmConnection(self): pass
            def listVehicles(self): return list(live.VEHICLES)
            def simListSceneObjects(self, regex): return []
            def reset(self): resets.append(1)
            def simPause(self, value): pass
            def simSetKinematics(self, *args, **kwargs): placements.append(kwargs["vehicle_name"])
            def simGetCollisionInfo(self, **kwargs):
                assert self.owner == threading.get_ident()
                return SimpleNamespace(has_collided=False, time_stamp=0)
            def hoverAsync(self, **kwargs): pass
            def ping(self): return True
        class Drone:
            def __init__(self, client, vehicle): self.client, self.vehicle = client, vehicle
            def enable_api_control(self, value): pass
            def arm(self): pass
            def hover(self): pass
            def get_raw_state(self):
                assert self.client.owner == threading.get_ident()
                return SimpleNamespace(position=first.position, quaternion=first.orientation,
                    velocity=np.zeros(3), acceleration=np.zeros(3), timestamp_ns=int(time.perf_counter()*1e9))
        class Planner:
            def __init__(self, layout):
                self.trajectory, self.config = SimpleNamespace(duration=.15), SimpleNamespace()
            def reference_window_for_trajectory(self, *args, **kwargs): return [first]*21
        class Pool:
            parameter_bytes = 0
            norm = {"x_mean": np.zeros(237)}
            def __init__(self, *args): pass
            def predict(self, *args): return np.array([9.8, 0., 0., 0.]), {"service_ms": .1}
            def runtime_info(self): return {}
        def send(drone, applied, calibration, duration):
            commands.append((drone.vehicle, duration))
            time.sleep(.001)
        value = plan()
        # Four real mock dispatches per drone, shorter than the planned flight;
        # the report must NOT accept this as a completed experiment.
        value.releases = lambda *args: ((k, time.perf_counter(), dict(step)) for k, step in enumerate(value.steps))
        args = SimpleNamespace(mode="dnn", model="plain4x256", device="cpu", phase="staggered",
            period_ms=50., deadline_ms=50., environment_id=48, route_index=0, host="localhost", port=41451,
            keep_gates=False, preprocess_policy="parallel", switch_plan=value,
            switch_runtime={}, switch_plan_sha256="plan", switch_profile_sha256="profile")
        from controller_architecture_study.state_budget_shadow import DEFAULT_CONFIG
        args.state_budget_config = DEFAULT_CONFIG
        args.joint_selection_shadow = True
        args.joint_profile = {"runtime": {}, "provenance": {}, "summary_sha256": "profile", "quality_scope": "mock route prior",
            "initial_producer_p95_ms": 1., "initial_dispatch_p95_ms": 1.,
            "initial_service_p95_ms": {"plain4x256": .1, "plain4x512": .1},
            "initial_release_jitter_p95_ms": 0., "candidates": [
                {"case": f"{s['model']}_T{s['period_ms']}", "model": s["model"],
                 "period_ms": s["period_ms"], "tracking_mse_m2": float(i+1)} for i, s in enumerate(value.steps)]}
        layout = {"segments": [], "gates": [dict(index=i+1, center=[10.+10*i, 0., -4.],
                   orientation=[1., 0., 0., 0.]) for i in range(5)]}
        api = SimpleNamespace(MultirotorClient=Client, KinematicsState=SimpleNamespace,
                              Vector3r=lambda *v: v, Quaternionr=lambda *v: v)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            patches = {"provenance": lambda *_: {}, "ModelPool": Pool, "load_layout": lambda *_: (layout, "route"),
                "StoredRoutePlanner": Planner, "new_output": lambda *_: Path(directory), "load_airsim_client": lambda: None,
                "require_airsim": lambda: api, "AirSimDrone": Drone, "vehicle_local_to_world_offset": lambda *_: np.zeros(3),
                "spawn_gate_sequences": lambda *a, **k: SimpleNamespace(destroy=lambda: None),
                "load_calibration": lambda: None, "encode": lambda *_: np.zeros(237), "send_control": send,
                "_crossing_results": lambda *_: [(i, 0., 0., True) for i in range(5)]}
            for name, replacement in patches.items():
                stack.enter_context(patch.object(live, name, replacement))
            from controller_architecture_study.joint_selection_shadow import JointShadowMonitor
            observe = JointShadowMonitor.observe
            coordinator = threading.get_ident()
            def postflight_observe(*args, **kwargs):
                self.assertEqual(threading.get_ident(), coordinator)
                self.assertEqual(len(commands), 12)
                # Longer than T50: replay analysis must not delay a flight request.
                time.sleep(.06)
                return observe(*args, **kwargs)
            stack.enter_context(patch.object(JointShadowMonitor, "observe", postflight_observe))
            stack.enter_context(patch.object(common.InferenceService, "feedback_snapshot",
                side_effect=AssertionError("no inline resource snapshot during flight")))
            with self.assertRaises(SystemExit):
                live.run(args)
            import json
            rows = [json.loads(line) for line in (Path(directory)/"requests.jsonl").read_text().splitlines()]
            report = common.read_json(Path(directory)/"report.json")
            traces = common.read_json(Path(directory)/"flight_trace.json")
            budget = common.read_json(Path(directory)/"state_budget_report.json")
            self.assertEqual(budget["samples"], 12)
            self.assertEqual(budget["applicable_samples"], 0)
            self.assertFalse(budget["control_changed"])
            self.assertEqual(report["state_dependent_T_max"]["mode"], "causal_shadow_replay")
            joint = common.read_json(Path(directory)/"joint_shadow_report.json")
            self.assertEqual(joint["samples"], 12)
            self.assertEqual(joint["execution_mode"], "post_flight_causal_replay")
            self.assertFalse(joint["recommendations_applied"])
            self.assertEqual(joint["policies"]["adaptive"]["decisions"], 12)
            self.assertEqual(report["joint_selection_shadow"]["capture_ms"]["count"], 12)
            self.assertEqual(report["timing"]["output_deadline_misses"], 0)
            self.assertTrue(all("joint_shadow_state" in r and "joint_shadow_snapshot_s" in r for r in rows))
            self.assertTrue(all("joint_shadow_start_s" not in r and "joint_shadow_ready_s" not in r for r in rows))
            args.output_dir = Path(directory)/"connection_failure"
            with patch.object(Client, "listVehicles", side_effect=RuntimeError("RPC unavailable")):
                with self.assertRaises(SystemExit) as failed:
                    live.run(args)
                self.assertEqual(failed.exception.code, 1)
            failure = common.read_json(args.output_dir/"report.json")
            self.assertIn("RPC unavailable", failure["errors"][0])
            self.assertFalse(failure["acceptance_passed"])
        self.assertEqual(len(resets), 1)
        self.assertEqual(sorted(placements), sorted(live.VEHICLES))
        self.assertFalse(report["online_resource_feedback"])
        self.assertFalse(report["acceptance_passed"])
        self.assertEqual(report["model_switches"], 6)
        self.assertEqual(report["period_switches"], 9)
        for vehicle in live.VEHICLES:
            vr = [r for r in rows if r["vehicle"] == vehicle]
            self.assertEqual([r["period_ms"] for r in vr], [50., 75., 100., 50.])
            self.assertTrue(all(r["deadline_ms"] == r["command_duration_ms"] for r in vr))
            self.assertEqual([t for v, t in commands if v == vehicle], [.05, .075, .1, .05])
            self.assertTrue(all(b["reference_elapsed_s"] > a["reference_elapsed_s"]
                                for a, b in zip(traces[vehicle], traces[vehicle][1:])))


if __name__ == "__main__":
    unittest.main()
