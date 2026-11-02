"""Protocol failure cases and a continuously advancing mocked simulator for E1."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from controller_architecture_study import state_mode_probe as probe
from controller_architecture_study import state_mode_flight as live


def trace_rows(duration=1.):
    return [{"t": float(t), "wall_s": float(t), "position_error_m": .1,
        "velocity_error_mps": .2, "h_m": .6, "collision": False,
        "path_arc_m": 5*float(t), "reference_arc_m": 5*float(t)} for t in np.linspace(0., duration, 51)]


class ProtocolTests(unittest.TestCase):
    def test_metrics_require_window_progress_and_contiguous_recovery(self):
        rows = trace_rows()
        good = probe.recovery_metrics(rows, [], 1.)
        self.assertTrue(good["joint_success"])
        self.assertFalse(probe.recovery_metrics(rows[:10], [], 1.)["joint_success"])
        stopped = [{**r, "path_arc_m": 0.} for r in rows]
        self.assertFalse(probe.recovery_metrics(stopped, [], 1.)["control_success"])
        alternating = [{**r, "position_error_m": 1. if i % 5 == 0 else .1} for i, r in enumerate(rows)]
        self.assertFalse(probe.recovery_metrics(alternating, [], 1.)["recovery_criterion_passed"])
        violation = [{**r, "h_m": -.01 if i == 25 else .6} for i, r in enumerate(rows)]
        self.assertFalse(probe.recovery_metrics(violation, [], 1.)["joint_success"])
        timing = probe.recovery_metrics(rows, [{"timing_violation": True}], 1.)
        self.assertTrue(timing["control_success"])
        self.assertFalse(timing["joint_success"])

    def test_late_output_never_dispatches(self):
        row = {"target_s": .02, "deadline_s": .05}
        action = live.checked_dispatch(np.array([9.81, 4., 0., 0.]), row, .023, .008)
        self.assertEqual(action[1], 3.)
        soft_late = {"target_s": .02, "deadline_s": .05}
        live.checked_dispatch(np.array([9.81, 0., 0., 0.]), soft_late, .04, .008)
        self.assertTrue(soft_late["delivery_target_missed"])
        self.assertFalse(soft_late["deadline_missed"])
        self.assertIn("command_send_s", soft_late)
        late = {"target_s": .02, "deadline_s": .05}
        with self.assertRaises(live.TrialFailure):
            live.checked_dispatch(np.array([9.81, 0., 0., 0.]), late, .05, .008)
        self.assertTrue(late["timing_violation"])
        self.assertTrue(late["deadline_missed"])
        self.assertNotIn("command_send_s", late)

    def test_pairing_and_failed_trials_are_retained(self):
        cases = [{"id": f"c{i}"} for i in range(12)]
        modes = [{"name": f"m{i}", "effective_macs": i+1} for i in range(6)]
        trials = probe.make_trials(cases, modes, 3, 123)
        self.assertEqual(len(trials), 216)
        self.assertEqual(len({r["id"] for r in trials}), 216)
        self.assertEqual(trials, probe.make_trials(cases, modes, 3, 123))
        for start in range(0, 216, 6):
            self.assertEqual(len({r["case"] for r in trials[start:start+6]}), 1)
        failed = {"status": "finished", "trial": trials[0], "joint_success": False,
            "control_success": False, "timing_comparable": True, "error_kind": "flight"}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"attempt_001"
            folder.mkdir()
            (folder/"report.json").write_text(json.dumps(failed))
            self.assertEqual(probe.completed_report(Path(temp)), failed)
            plan = {"cases": cases, "modes": modes, "trials": trials, "repeats": 3}
            summary = probe.save_summary(Path(temp), plan, [failed])
            self.assertEqual(summary["completed_trials"], 1)
            self.assertTrue(all(r["status"] == "unknown" for r in summary["rows"]))
            root = ET.parse(Path(temp)/"recovery_heatmap.svg").getroot()
            self.assertTrue(any(node.text == "pending" for node in root.iter()))
            self.assertTrue(any(node.text == "0/1 I=0 partial" for node in root.iter()))

    def test_planned_states_really_have_requested_risk(self):
        from airsim_interface import RawDroneState
        from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
        from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow

        cases, layouts = probe.make_cases(3.)
        self.assertEqual(len(cases), 12)
        for case in cases:
            planner = StoredRoutePlanner(layouts[case["geometry"]])
            window = FastReferenceWindow(planner.trajectory, planner.config)
            state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
            features = probe.risk_features(state, window.window(case["reference_phase_s"])[0])
            self.assertAlmostEqual(features["h_m"], case["requested_h_m"])
            self.assertAlmostEqual(features["h_dot_mps"], case["requested_h_dot_mps"])

    def test_real_export_selection_avoids_worst_exits_and_includes_best(self):
        metrics = probe.read_json(probe.EXPORT/"offline_metrics.json")
        modes, rows = probe.select_modes(metrics)
        self.assertEqual(len(modes), 6)
        self.assertEqual(len({m["name"] for m in modes}), 6)
        self.assertGreater(len({m["depth"] for m in modes}), 1)
        self.assertGreater(len({m["width"] for m in modes}), 1)
        self.assertEqual(modes[-1]["validation_mse"], min(r["validation_mse"] for r in rows))
        self.assertLessEqual(max(r["validation_mse"] for r in modes), 10*modes[-1]["validation_mse"])

    def test_mock_flight_preserves_physics_and_matches_delivery_targets(self):
        from airsim_interface import RawDroneState

        coefficients = np.zeros((3, 6))
        coefficients[0, 1], coefficients[2, 0] = 5., -4.
        layout = {"duration": 8., "segments": [{"coefficients": coefficients.tolist(),
            "duration": 8., "start_time": 0., "heading_orientation": [1., 0., 0., 0.]}],
            "gates": [{"index": 1, "center": [10., 0., -4.], "orientation": [1., 0., 0., 0.]}]}
        case = {"position": [0., 0., -4.], "velocity": [5., 0., 0.], "quaternion": [1., 0., 0., 0.],
            "held_control": [9.81, 0., 0., 0.], "reference_phase_s": 0.}
        clock = SimpleNamespace(now=10., origin=10., paused=True)
        pauses, sends = [], []

        def tick(dt):
            clock.now += max(float(dt), .000001)

        class Client:
            rpc_delay = .011
            def reset(self):
                pass
            def simPause(self, value):
                pauses.append(value)
                clock.paused = value
                if not value:
                    clock.origin = clock.now
            def simSetKinematics(self, kin, **kwargs):
                self_outer.assertTrue(kwargs["ignore_collision"])
                self.kin = kin
            def simSetObjectPose(self, name, pose, teleport):
                self_outer.assertTrue(teleport)
                self.actor_position = pose.position_val.copy()
                self.actor_orientation = pose.orientation_val.copy()
                return True
            def simGetVehiclePose(self, **kwargs):
                p, q = self.actor_position, self.actor_orientation
                return SimpleNamespace(position=SimpleNamespace(x_val=p[0], y_val=p[1], z_val=p[2]),
                    orientation=SimpleNamespace(w_val=q[0], x_val=q[1], y_val=q[2], z_val=q[3]))
            def simGetCollisionInfo(self, **kwargs):
                tick(.0001)
                return SimpleNamespace(time_stamp=0, has_collided=False, object_name="")
            def ping(self):
                tick(self.rpc_delay)
        self_outer = self
        client = Client()

        class Drone:
            def enable_api_control(self, enabled):
                pass
            def arm(self):
                pass
            def hover(self):
                pass
            def get_raw_state(self):
                tick(.0001)
                t = 0. if clock.paused else clock.now-clock.origin
                return RawDroneState(client.kin.position+[5.*t, 0., 0.], client.kin.orientation,
                    client.kin.linear_velocity, np.zeros(3), np.zeros(3), int(1e9+t*1e9))

        class Pool:
            lag = .002
            def predict(self, name, feature):
                tick(self.lag)
                return np.array([9.81, 0., 0., 0.]), {"inference_ms": self.lag*1000}

        api = SimpleNamespace(KinematicsState=SimpleNamespace, Pose=SimpleNamespace, Vector3r=lambda *v: np.array(v),
            Quaternionr=lambda x, y, z, w: np.array([w, x, y, z]))
        args = SimpleNamespace(vehicle="Drone1", duration=1., period_ms=50., delivery_ms=20., delivery_tolerance_ms=8.)
        trial = {"id": "mock", "mode": "d11_w0256", "case": "c", "repeat": 0}
        with tempfile.TemporaryDirectory() as temp, \
                patch("airsim_interface.require_airsim", return_value=api), \
                patch("airsim_interface.AirSimDrone", return_value=Drone()), \
                patch("airsim_interface.vehicle_local_to_world_offset", return_value=np.zeros(3)), \
                patch("gate_visualization.spawn_gate_sequences", return_value=SimpleNamespace(destroy=lambda: None)), \
                patch("control_adapter.send_control", side_effect=lambda *a, **kw: sends.append(clock.now)), \
                patch.object(live.time, "perf_counter", side_effect=lambda: clock.now), \
                patch.object(live.time, "sleep", side_effect=tick):
            good = live.flight(client, Pool(), layout, case, trial, args, Path(temp))
            commands = probe.read_json(Path(temp)/"commands.json")
            self.assertTrue(good["joint_success"], good)
            self.assertEqual(pauses, [True, False, False])
            self.assertGreater(len(commands), 15)
            for command in commands:
                self.assertGreaterEqual(command["command_send_s"], command["target_s"])
                self.assertLess(command["command_send_lateness_ms"], .5)
                self.assertGreater(command["rpc_return_s"], command["target_s"]+.008)
                self.assertFalse(command["timing_violation"])
            placement = probe.read_json(Path(temp)/"initial_state.json")
            self.assertTrue(placement["placement_teleport"])
            self.assertEqual(placement["actor_readback_position_rotation"], [0., 0.])
            # A soft target miss keeps collecting, without masquerading as comparable.
            client.rpc_delay = .0001
            Pool.lag = .04
            soft_dir = Path(temp)/"soft"
            soft_dir.mkdir()
            soft = live.flight(client, Pool(), layout, case, trial, args, soft_dir)
            self.assertIsNone(soft["error"], soft)
            self.assertTrue(soft["window_covered"])
            self.assertTrue(soft["control_success"])
            self.assertFalse(soft["timing_comparable"])
            self.assertGreater(soft["delivery_deviations"], 0)
            self.assertEqual(soft["deadline_misses"], 0)
            # A return after D is uncertain evidence, not a retroactive send failure.
            Pool.lag, client.rpc_delay = .002, .04
            uncertain_dir = Path(temp)/"uncertain_rpc"
            uncertain_dir.mkdir()
            uncertain = live.flight(client, Pool(), layout, case, trial, args, uncertain_dir)
            self.assertIsNone(uncertain["error"], uncertain)
            self.assertTrue(uncertain["control_success"])
            self.assertFalse(uncertain["timing_comparable"])
            self.assertGreater(uncertain["rpc_returns_after_deadline"], 0)
            self.assertEqual(uncertain["deadline_misses"], 0)
            client.rpc_delay = .0001
            initial_count = len(sends)
            Pool.lag = .06
            late_dir = Path(temp)/"late"
            late_dir.mkdir()
            late = live.flight(client, Pool(), layout, case, trial, args, late_dir)
            self.assertEqual(late["error_kind"], "timing")
            self.assertFalse(late["joint_success"])
            self.assertEqual(len(sends), initial_count+1)  # old held command only

    def test_actor_placement_prevents_swept_crossing_of_previous_gate(self):
        # Regression for the observed birth-to-turn move through Gate1's bottom bar.
        from airsim_interface import RawDroneState
        from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
        from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow

        cases, layouts = probe.make_cases(3.)
        case = next(c for c in cases if c["id"] == "turn_h0.6_v0.4")
        offset = np.array([0., -3., .14649587869644165])
        target = np.asarray(case["position"])
        gate = layouts["turn"]["gates"][0]
        w, x, y, z = gate["orientation"]
        normal = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
        fraction = np.dot(np.asarray(gate["center"])-offset, normal)/np.dot(target-offset, normal)
        intersection = offset+fraction*(target-offset)
        self.assertGreater(fraction, 0.)
        self.assertLess(fraction, 1.)
        self.assertLess(abs(intersection[2]-(gate["center"][2]+2.)), .075)
        # Actual warmup must encode real case states, not the normalization mean.
        features = probe.warmup_features(cases, layouts)
        self.assertEqual(set(features), {c["id"] for c in cases})
        self.assertTrue(all(v.shape == (237,) for v in features.values()))
        planner = StoredRoutePlanner(layouts["turn"])
        reference = FastReferenceWindow(planner.trajectory, planner.config).window(case["reference_phase_s"])[0]
        state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(probe.risk_features(state, reference)["h_m"], .6)


if __name__ == "__main__":
    unittest.main()
