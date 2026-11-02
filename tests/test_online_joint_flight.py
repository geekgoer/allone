"""Online next-request switching with CPU/mock interfaces only; no real flight."""

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
from controller_architecture_study import online_joint_flight as online
from controller_architecture_study import probe_airsim_three as live


def profile():
    return {"runtime": {}, "summary_sha256": "mock", "quality_scope": "mock route prior",
        "initial_producer_p95_ms": 1., "initial_dispatch_p95_ms": 1.,
        "initial_service_p95_ms": {"plain4x256": .1, "plain4x512": .1},
        "initial_release_jitter_p95_ms": 0., "candidates": [
            {"case": f"{m}_T{t}", "model": m, "period_ms": float(t), "tracking_mse_m2": float(i+1)}
            for i, (m, t) in enumerate([("plain4x256", 50), ("plain4x512", 75), ("plain4x256", 100)])]}


def config():
    return {**online.load_config(online.DEFAULT_CONFIG), "response_floor_ms": 1., "response_margin_ms": 0.}


class SelectorTests(unittest.TestCase):
    def test_state_age_uses_committed_interval_not_next_period(self):
        value = online.StateResourceSelector(profile(), config()).select(3, 0., [],
            dict(applicable=True, observed_rate_bound_exceeded=False, validity_budget_ms=80.),
            committed_period_ms=100.)
        self.assertIsNone(value["recommendation"])
        self.assertTrue(all(not c["state_feasible"] for c in value["candidate_options"]))

    def test_larger_next_T_cannot_extend_current_deadline(self):
        result = online.StateResourceSelector(profile(), config()).select(3, .06, [],
            dict(applicable=False, observed_rate_bound_exceeded=False),
            release_elapsed_ms=60., committed_period_ms=50.)
        self.assertIsNone(result["recommendation"])
        self.assertTrue(all(not c["next_release_not_elapsed"] for c in result["candidate_options"]))

    def exercise(self, gates=(), timestamps=(.001, .002, .003, .006, .007), snapshot=None):
        recorded = []
        service = SimpleNamespace(feedback_snapshot=lambda _: snapshot or
            {"recent": [], "pending_model_counts": {}, "inference_error": None}, record_abort=recorded.append)
        controller = online.OnlineJointController(profile(), config(), service, ["a"])
        row = dict(vehicle="a", cycle=0, scheduled_s=0., sample_start_s=0., model="plain4x256",
                   period_ms=50., deadline_ms=50.)
        raw = SimpleNamespace(position=np.array([0., 0., -4.]), velocity=np.zeros(3))
        return controller, service, row, raw, recorded, patch.object(online.time, "perf_counter", side_effect=timestamps)

    def test_future_completion_and_dispatch_never_inform_current_choice(self):
        snapshot = {"recent": [dict(model="plain4x256", status="ready", ready_s=1.,
                    service_ms=1000., sample_start_s=0., scheduled_s=0.)]*8,
                    "pending_model_counts": {}, "inference_error": None}
        controller, _, row, raw, recorded, clock = self.exercise(snapshot=snapshot)
        controller.history.append(dict(dnn_dispatch_s=1., sample_start_s=0., enqueue_s=0., ready_s=.9))
        with clock:
            controller.select_next(row, raw, [], live.GATE)
        self.assertEqual(recorded, [])
        self.assertEqual(row["next_model"], "plain4x256")
        self.assertEqual(row["joint_online_decision"]["completed_response_reserve_ms"], 1.)
        self.assertEqual(row["joint_online_decision"]["service_p95_ms"]["plain4x256"], .1)
        self.assertGreater(row["joint_online_admission"]["selection_tail_ms"], 0.)

    def test_no_candidate_is_logged_without_inference_or_deadline_extension(self):
        from environment import Gate
        controller, _, row, raw, recorded, clock = self.exercise()
        raw.position = np.array([0., 3., -4.])
        gate = Gate(1, [0., 0., -4.], [1., 0., 0., 0.])
        with clock, self.assertRaisesRegex(RuntimeError, "no eligible next"):
            controller.select_next(row, raw, [gate], live.GATE)
        self.assertEqual(recorded, [row])
        self.assertEqual(row["deadline_ms"], 50.)
        self.assertNotIn("enqueue_s", row)
        self.assertEqual(row["joint_online_sample"]["position"], [0., 3., -4.])
        self.assertNotIn("a", controller.next_choices)

    def test_slow_selection_is_rejected_and_its_time_is_retained(self):
        controller, _, row, raw, recorded, clock = self.exercise(timestamps=(.001, .002, .003, .060, .061))
        with clock, self.assertRaisesRegex(RuntimeError, "immutable current deadline"):
            controller.select_next(row, raw, [], live.GATE)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(row["period_ms"], row["deadline_ms"])
        self.assertAlmostEqual(1000*(row["joint_online_ready_s"]-row["joint_online_start_s"]), 59.)

    def test_late_dispatch_cannot_commit_next_choice(self):
        controller, _, row, _, _, _ = self.exercise()
        row.update(dnn_dispatch_s=.051, next_model="plain4x512", next_period_ms=75., next_deadline_ms=75.)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            controller.commit_dispatch(row)
        self.assertFalse(controller.history)
        self.assertFalse(controller.next_choices)

    def borderline_selection(self):
        value = profile()
        value.update(initial_producer_p95_ms=28.063099948, initial_dispatch_p95_ms=.363824976,
                     initial_release_jitter_p95_ms=14.632366685,
                     initial_service_p95_ms={"plain4x256": 1.90669999, "plain4x512": 3.25})
        settings = {**config(), "response_floor_ms": 30., "response_margin_ms": 5.}
        clock = (.0296844, .0297, .0298375, .0308405, .03085)
        with patch(__name__+".profile", return_value=value), patch(__name__+".config", return_value=settings):
            controller, _, row, raw, recorded, timer = self.exercise(timestamps=clock)
        row["sample_start_s"] = .0017744
        return controller, row, raw, recorded, timer

    def test_selection_cost_rechecks_all_next_candidates_before_aborting(self):
        # Recorded cycle 71: T50 fits by .034ms before the 1.003ms selector tail.
        controller, row, raw, recorded, timer = self.borderline_selection()
        with timer:
            controller.select_next(row, raw, [], live.GATE)
        self.assertEqual(recorded, [])
        self.assertEqual(row["next_period_ms"], 75.)
        self.assertEqual(row["next_model"], "plain4x512")
        self.assertEqual(row["period_ms"], row["deadline_ms"])
        self.assertEqual(row["period_ms"], 50.)
        self.assertLess(row["joint_online_ready_s"], .05)
        self.assertIsNone(controller.selectors["a"].current)  # Not committed yet.
        options = row["joint_online_decision"]["candidate_options"]
        self.assertFalse(options[0]["estimated_feasible"])
        self.assertTrue(options[1]["estimated_feasible"])
        self.assertTrue(options[2]["estimated_feasible"])

    def test_longer_next_period_cannot_override_gate_validity_budget(self):
        controller, row, raw, recorded, timer = self.borderline_selection()
        state = dict(applicable=True, observed_rate_bound_exceeded=False, validity_budget_ms=100.5)
        with timer, patch.object(online, "gate_budget", return_value=state), \
                self.assertRaisesRegex(RuntimeError, "no eligible next"):
            controller.select_next(row, raw, [], live.GATE)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(row["deadline_ms"], 50.)
        self.assertFalse(controller.next_choices)

    def test_rescreen_time_is_charged_and_can_require_another_candidate(self):
        controller, row, raw, recorded, _ = self.borderline_selection()
        with patch.object(online.time, "perf_counter", side_effect=
                          (.0296844, .0297, .0298375, .02984, .03084, .0309)):
            controller.select_next(row, raw, [], live.GATE)
        self.assertEqual(recorded, [])
        self.assertEqual(row["next_period_ms"], 75.)
        self.assertEqual(row["joint_online_admission"]["rescreens"], 2)
        self.assertAlmostEqual(row["joint_online_admission"]["selection_tail_ms"], 1.0625)
        self.assertAlmostEqual(row["joint_online_decision"]["charged_selection_tail_ms"], 1.0025)

    def test_preview_and_failed_dispatch_do_not_reset_committed_dwell(self):
        controller, row, raw, _, timer = self.borderline_selection()
        selector = controller.selectors["a"]
        selector.current, selector.last_switch = "plain4x256_T50", 0.
        with timer:
            controller.select_next(row, raw, [], live.GATE)
        # Leave an infeasible T50 immediately, despite the dwell rule.
        self.assertEqual(row["next_period_ms"], 75.)
        self.assertEqual(selector.current, "plain4x256_T50")
        row.update(enqueue_s=.031, ready_s=.032, dnn_dispatch_s=.051)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            controller.commit_dispatch(row)
        self.assertEqual(selector.current, "plain4x256_T50")
        self.assertEqual(selector.last_switch, 0.)
        row["dnn_dispatch_s"] = .033
        controller.commit_dispatch(row)
        self.assertEqual(selector.current, "plain4x512_T75")
        self.assertEqual(selector.last_switch, .033)


class DispatchGuardTests(unittest.TestCase):
    def test_peer_stop_and_real_deadline_expiry_are_distinct(self):
        for stopped, now, outcome in [(False, .04, None), (True, .04, "peer_abort"),
                                      (False, .05, "late_output_not_applied"),
                                      (True, .06, "late_output_not_applied")]:
            with self.subTest(stopped=stopped, now=now):
                stop, row = threading.Event(), dict(status="ready", scheduled_s=0.,
                    sample_start_s=0., enqueue_s=0., start_s=0., ready_s=.01,
                    service_ms=1., deadline_ms=50.)
                if stopped:
                    stop.set()
                with patch.object(live.time, "perf_counter", return_value=now):
                    if outcome is None:
                        live.check_online_dispatch(row, stop, .05)
                    else:
                        with self.assertRaises(RuntimeError):
                            live.check_online_dispatch(row, stop, .05)
                self.assertEqual(row.get("control_outcome"), outcome)
                self.assertEqual(row["dispatch_guard_s"], now)
                self.assertEqual(row["dispatch_guard_stop_requested"], stopped)
                self.assertEqual(row["dispatch_guard_deadline_expired"], now >= .05)
                stats = common.summarize_timing([row], 50.)
                self.assertEqual(stats["control_deadline_aborts"], int(now >= .05))
                self.assertEqual(stats["peer_aborts"], int(outcome == "peer_abort"))


class ScheduleTests(unittest.TestCase):
    def rows(self):
        pairs = [("plain4x256", 50.), ("plain4x512", 75.), ("plain4x256", 100.)]
        elapsed, rows = 0., []
        for cycle, (m, t) in enumerate(pairs):
            following = pairs[(cycle+1) % len(pairs)]
            rows.append(dict(vehicle="a", cycle=cycle, scheduled_s=10.+elapsed, model=m,
                period_ms=t, deadline_ms=t, command_duration_ms=t, next_choice_committed=True,
                next_model=following[0], next_period_ms=following[1], next_deadline_ms=following[1]))
            elapsed += t/1000
        return rows

    def test_release_transition_and_horizon_are_checked(self):
        rows, initial = self.rows(), dict(model="plain4x256", period_ms=50., deadline_ms=50.)
        self.assertTrue(online.audit_schedule(rows, ["a"], 10., .2, "sync", initial)["passed"])
        for bad in (rows[:-1], rows+[dict(rows[-1])],
                    [{**rows[0], "deadline_ms": 100.}]+rows[1:],
                    rows[:-1]+[{**rows[-1], "next_deadline_ms": 200.}],
                    [rows[0], {**rows[1], "scheduled_s": 10.075}, rows[2]]):
            self.assertFalse(online.audit_schedule(bad, ["a"], 10., .2, "sync", initial)["passed"])

    def test_empty_or_incomplete_schedule_cannot_be_accepted(self):
        initial = dict(model="plain4x256", period_ms=50., deadline_ms=50.)
        self.assertFalse(online.audit_schedule([], ["a"], None, .2, "sync", initial)["passed"])
        row = dict(vehicle="a", status="ready", scheduled_s=1., sample_start_s=1.,
            enqueue_s=1., start_s=1., ready_s=1.001, service_ms=1., deadline_ms=50., dnn_dispatch_s=1.002)
        with tempfile.TemporaryDirectory() as directory:
            report = common.save_timing(Path(directory), [row], None,
                dict(status="COMPLETE", planned_releases=1, all_15_crossings_passed=True,
                     schedule_accounting_passed=False))
        self.assertFalse(report["acceptance_passed"])


class MockFlightTests(unittest.TestCase):
    def run_flight(self, directory, *, reject=False, late=False):
        commands, hovers, resets, calls = [], [], [], {}
        raw_position, quaternion = np.array([0., 0., -4.]), np.array([1., 0., 0., 0.])
        reference = SimpleNamespace(position=raw_position, orientation=quaternion)
        class Client:
            def __init__(self, **kwargs): self.owner = threading.get_ident()
            def confirmConnection(self): pass
            def listVehicles(self): return list(live.VEHICLES)
            def simListSceneObjects(self, regex): return []
            def reset(self): resets.append(1)
            def simPause(self, value): pass
            def simSetKinematics(self, *args, **kwargs): pass
            def simGetCollisionInfo(self, **kwargs):
                assert self.owner == threading.get_ident()
                return SimpleNamespace(has_collided=False, time_stamp=0)
            def hoverAsync(self, **kwargs): pass
            def ping(self): pass
        class Drone:
            def __init__(self, c, vehicle): self.client, self.vehicle = c, vehicle
            def enable_api_control(self, value): pass
            def arm(self): pass
            def hover(self): hovers.append(self.vehicle)
            def get_raw_state(self):
                assert self.client.owner == threading.get_ident()
                return SimpleNamespace(position=raw_position, quaternion=quaternion, velocity=np.zeros(3),
                    acceleration=np.zeros(3), timestamp_ns=int(time.perf_counter()*1e9))
        class Planner:
            def __init__(self, layout): self.trajectory, self.config = SimpleNamespace(duration=.15), SimpleNamespace()
            def reference_window_for_trajectory(self, *a, **k): return [reference]*21
        class Pool:
            parameter_bytes, norm = 0, {"x_mean": np.zeros(237)}
            def __init__(self, *args): pass
            def runtime_info(self): return {}
            def predict(self, *args): return np.array([9.8, 0., 0., 0.]), {"service_ms": .1}
        def send(drone, applied, calibration, duration):
            commands.append((drone.vehicle, duration))
            if late:
                time.sleep(.065)
        actual_recommend = online.StateResourceSelector._recommend
        def recommend(policy, decision, now, *, commit):
            result = actual_recommend(policy, decision, now, commit=commit)
            if reject:
                return {**result, "recommendation": None, "status": "BUDGET_EXHAUSTED"}
            if "charged_selection_tail_ms" not in decision:
                calls[id(policy)] = calls.get(id(policy), -1)+1
            index = calls[id(policy)]
            chosen = result["candidate_options"][(index+1) % 3]
            self.assertTrue(chosen["estimated_feasible"])
            return {**result, "recommendation": chosen}
        args = SimpleNamespace(mode="dnn", model="plain4x256", device="cpu", phase="staggered",
            period_ms=50., deadline_ms=50., environment_id=48, route_index=0, host="localhost", port=41451,
            keep_gates=False, preprocess_policy="vectorized", joint_online=True, policy="adaptive",
            joint_profile=profile(), state_budget_config=online.DEFAULT_CONFIG, output_dir=Path(directory)/"flight")
        layout = {"segments": [], "gates": [dict(index=i+1, center=[10.+10*i, 0., -4.],
                   orientation=quaternion.tolist()) for i in range(5)]}
        api = SimpleNamespace(MultirotorClient=Client, KinematicsState=SimpleNamespace,
                              Vector3r=lambda *v: v, Quaternionr=lambda *v: v)
        with ExitStack() as stack:
            patches = {"provenance": lambda *_: {}, "ModelPool": Pool, "load_layout": lambda *_: (layout, "mock"),
                "StoredRoutePlanner": Planner, "load_airsim_client": lambda: None, "require_airsim": lambda: api,
                "AirSimDrone": Drone, "vehicle_local_to_world_offset": lambda *_: np.zeros(3),
                "spawn_gate_sequences": lambda *a, **k: SimpleNamespace(destroy=lambda: None),
                "load_calibration": lambda: None, "fast_encode": lambda *_: np.zeros(237),
                "FastReferenceWindow": lambda *_: SimpleNamespace(window=lambda *_: [reference]*21),
                "send_control": send, "_crossing_results": lambda *_: [(i, 0., 0., True) for i in range(5)],
                "ConvexTeacher": lambda: self.fail("no MPC may be constructed in online DNN control")}
            for name, replacement in patches.items():
                stack.enter_context(patch.object(live, name, replacement))
            # Shorten only mock flight duration; production duration/CLI is unchanged.
            stack.enter_context(patch.object(live, "max", lambda a, b: .22 if a == 20. else max(a, b), create=True))
            stack.enter_context(patch.object(online.StateResourceSelector, "_recommend", recommend))
            stack.enter_context(patch("controller_architecture_study.joint_controller_sweep.time_mean_squared_error", return_value=.01))
            try:
                live.run(args)
            except SystemExit:
                pass
        report = common.read_json(args.output_dir/"report.json")
        import json
        rows = [json.loads(line) for line in (args.output_dir/"requests.jsonl").read_text().splitlines()]
        return report, rows, commands, hovers, resets

    def test_online_choices_really_change_next_model_and_T_without_rephasing(self):
        with tempfile.TemporaryDirectory() as directory:
            report, rows, commands, _, resets = self.run_flight(directory)
        self.assertTrue(report["acceptance_passed"], report["errors"])
        self.assertEqual(resets, [1])
        self.assertEqual(report["model_switches"], 6)
        self.assertEqual(report["period_switches"], 6)
        self.assertEqual(report["timing"]["online_selection_ms"]["count"], 9)
        for vehicle in live.VEHICLES:
            records = [r for r in rows if r["vehicle"] == vehicle]
            self.assertEqual([r["model"] for r in records], ["plain4x256", "plain4x512", "plain4x256"])
            self.assertEqual([r["period_ms"] for r in records], [50., 75., 100.])
            self.assertTrue(all(r["period_ms"] == r["deadline_ms"] == r["command_duration_ms"] for r in records))
            np.testing.assert_allclose(np.diff([r["scheduled_s"] for r in records]), [.05, .075], atol=1e-6)
            self.assertEqual([t for v, t in commands if v == vehicle], [.05, .075, .1])

    def test_no_candidate_aborts_before_any_DNN_command_and_keeps_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report, rows, commands, hovers, _ = self.run_flight(directory, reject=True)
        self.assertFalse(report["acceptance_passed"])
        self.assertEqual(commands, [])
        self.assertGreater(report["timing"]["policy_aborts"], 0)
        self.assertGreaterEqual(len(hovers), 3)
        self.assertTrue(all(r.get("status") == "policy_abort" for r in rows))

    def test_late_async_dispatch_is_not_hidden_or_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            report, rows, _, _, _ = self.run_flight(directory, late=True)
        self.assertFalse(report["acceptance_passed"])
        self.assertGreater(report["timing"]["dnn_dispatch_deadline_misses"], 0)
        self.assertFalse(any(r.get("next_choice_committed") for r in rows))


if __name__ == "__main__":
    unittest.main()
