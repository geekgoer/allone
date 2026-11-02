from types import SimpleNamespace
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from controller_architecture_study.safety_comparison_core import (LayerJob, SegmentServer, bounded_release,
    _process_worker, full_job_cost_profile, joint_release_plan, paper_score, plan_after_delivery, require_startup_vehicles, safe_horizon)
from controller_architecture_study.run_safety_comparison import clean_split, config, scenarios, score
from environment import Gate


class SafetyComparisonTests(unittest.TestCase):
    def test_ipc_worker_does_not_add_timed_pipe_wait_before_delivering_ready_output(self):
        connection = Mock()
        def immediate_poll(timeout=0.):
            self.assertEqual(timeout,0., "Windows Pipe timed waits can round to a15.6ms tick")
            return True
        connection.poll.side_effect = immediate_poll
        connection.recv.return_value = ("stop",)
        pool = SimpleNamespace(profile={}, provenance=lambda: {})
        server = Mock()
        server.drain.return_value = []
        server.segments = []
        with patch("controller_architecture_study.safety_comparison_core.LayerPool",return_value=pool), \
                patch("controller_architecture_study.safety_comparison_core.SegmentServer",return_value=server):
            _process_worker(connection,"framework")
        self.assertEqual([c.args[0][0] for c in connection.send.call_args_list],["ready","closed"])

    def test_runtime_added_vehicles_are_rejected_even_if_list_vehicles_contains_them(self):
        client = Mock()
        client.getSettingsString.return_value = json.dumps({"Vehicles": {"Drone1": {}}})
        client.listVehicles.return_value = ["Drone1", "Drone2"]
        with self.assertRaisesRegex(RuntimeError, "startup settings"):
            require_startup_vehicles(client, ["Drone1", "Drone2"])
        client.reset.assert_not_called()
        client.simAddVehicle.assert_not_called()
        settings = Path(__file__).parents[1]/"controller_architecture_study/configs/airsim_safety_12.json"
        client.getSettingsString.return_value = settings.read_text(encoding="utf-8")
        client.listVehicles.return_value = [f"Drone{i+1}" for i in range(12)]
        require_startup_vehicles(client, client.listVehicles.return_value)

    def test_shared_server_preempts_only_between_segments_under_both_policies(self):
        for policy in ("static", "framework"):
            entered, resume = threading.Event(), threading.Event()

            class Job:
                def __init__(self, name):
                    self.name, self.cursor, self.done, self.output = name, 0, False, None

                def step(self):
                    if self.name == "A" and self.cursor == 0:
                        entered.set()
                        if not resume.wait(2.):
                            raise RuntimeError("test did not release first segment")
                    self.cursor += 1
                    self.done = self.cursor == 3
                    if self.done:
                        self.output = np.ones(4)

            warmed = []
            pool = SimpleNamespace(start=lambda name, _: Job(name),
                warm_worker=lambda: warmed.append(threading.current_thread().name))
            server = SegmentServer(pool, policy)
            self.assertEqual(warmed, ["safety-shared-CPU"])
            try:
                now = time.perf_counter()
                server.submit(None, {"vehicle":"A", "model":"A", "sample_id":"A:0", "priority":1, "deadline_s":now+10.})
                self.assertTrue(entered.wait(2.))
                server.submit(None, {"vehicle":"B", "model":"B", "sample_id":"B:0", "priority":0, "deadline_s":now+5.})
                resume.set()
                finished = []
                until = time.perf_counter()+2.
                while len(finished) < 2 and time.perf_counter() < until:
                    finished += server.drain()
                    time.sleep(.001)
                self.assertEqual(len(finished), 2)
                self.assertEqual([s["vehicle"] for s in server.segments], ["A", "B", "B", "B", "A", "A"])
                self.assertEqual([s["layer_start"] for s in server.segments if s["vehicle"] == "A"], [0,1,2])
            finally:
                resume.set()
                server.close()

    def test_segmentation_keeps_exact_function_and_independent_contexts(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(237, 16), torch.nn.ReLU(), torch.nn.Linear(16,16),
            torch.nn.ReLU(), torch.nn.Linear(16,16), torch.nn.ReLU(), torch.nn.Linear(16,4)).eval()
        norm = {"x_mean": np.zeros(237, np.float32), "x_std": np.ones(237, np.float32),
            "u_mean": np.array([9.,0.,0.,0.], np.float32), "u_std": np.ones(4,np.float32)}
        layers = tuple(model.children())
        chunks = tuple(torch.jit.script(torch.nn.Sequential(*layers[i:i+4])) for i in range(0,len(layers),4))
        pool = SimpleNamespace(torch=torch, norm=norm, layers={"m": layers}, chunks={"m": chunks})
        x = np.random.default_rng(0).normal(size=(2,237)).astype(np.float32)
        jobs = [LayerJob(pool,"m",row) for row in x]
        jobs[0].step()
        jobs[1].step()
        self.assertFalse(jobs[0].done)
        for job in reversed(jobs):
            job.step()
        with torch.inference_mode():
            for job, row in zip(jobs,x):
                expected = model(torch.from_numpy(row[None])).numpy()[0]+norm["u_mean"]
                np.testing.assert_array_equal(job.output, expected)
                with self.assertRaises(ValueError):
                    job.step()

    def test_bounded_planner_protects_other_commitment_and_does_not_extend_deadline(self):
        committed = [{"vehicle":"other", "deadline_s":.010, "service_bound_s":.006, "planned_release_s":0.}]
        release, feasible, count = bounded_release(0., .020, .006, committed, "own")
        self.assertTrue(feasible)
        self.assertLessEqual(release+.006,.020)
        self.assertLessEqual(count,3)
        release, feasible, _ = bounded_release(0., .008, .006, committed, "own")
        self.assertFalse(feasible)
        self.assertEqual(release,0.)

    def test_delivery_planning_uses_post_rpc_clock_and_full_job_cost(self):
        row = {"vehicle":"own", "command_send_s":10., "rpc_return_s":10.04, "service_bound_s":.020}
        slot, decision = plan_after_delivery(row, {"sample_s":10., "budget_s":.050}, [])
        self.assertEqual(decision["planning_time_s"], 10.04)
        self.assertAlmostEqual(slot["deadline_s"],10.05)
        self.assertGreaterEqual(slot["planned_release_s"],row["rpc_return_s"])
        self.assertFalse(decision["estimated_feasible"])
        committed = [{"vehicle":str(i),"planned_release_s":0.,"deadline_s":.025,"service_bound_s":.010} for i in range(2)]
        self.assertFalse(bounded_release(0.,.025,.010,committed,"own")[1])

    def test_cost_profile_includes_frontend_and_command_not_only_DNN(self):
        rows = [{"model":"m", "status":"applied", "job_started_s":1., "enqueue_s":1.004,
            "ready_s":1.005, "rpc_return_s":1.015} for _ in range(20)]
        profile = full_job_cost_profile(rows)["m"]
        self.assertAlmostEqual(profile["p95_s"],.015)
        self.assertAlmostEqual(profile["estimated_cost_s"],.02075)
        self.assertFalse(profile["is_WCET"])

    def test_joint_plan_pulls_old_release_forward_for_a_new_job_without_moving_deadlines(self):
        slots = {"old":{"vehicle":"old", "planned_release_s":.1295,"deadline_s":.150,"service_bound_s":.0175},
            "new":{"vehicle":"new", "planned_release_s":.124,"deadline_s":.160,"service_bound_s":.0175}}
        result, info = joint_release_plan(.124, slots)
        self.assertTrue(info["estimated_feasible"])
        self.assertLess(result["old"]["planned_release_s"],slots["old"]["planned_release_s"])
        self.assertEqual(info["edf_order"],["old","new"])
        for v in slots:
            self.assertEqual(result[v]["deadline_s"],slots[v]["deadline_s"])
            self.assertLessEqual(result[v]["planned_release_s"]+result[v]["service_bound_s"],result[v]["deadline_s"])
        overloaded = {v:{**s,"service_bound_s":.050} for v,s in slots.items()}
        result,info = joint_release_plan(.124, overloaded)
        self.assertFalse(info["estimated_feasible"])
        self.assertEqual(info["collective_idle_s"],0.)
        self.assertEqual(result["old"]["planned_release_s"],.124)

    def test_aperture_means_clear_width_and_risk_is_not_constant(self):
        gate=Gate(3,np.array([10.,0.,-4.]),np.array([1.,0.,0.,0.]))
        wide,_=safe_horizon([9.,0.,-4.],[5.,0.,0.],[gate],2.)
        near,_=safe_horizon([9.,.40,-4.],[5.,1.,0.],[gate],2.)
        narrow,_=safe_horizon([9.,0.,-4.],[5.,0.,0.],[gate],1.5)
        self.assertGreater(wide,near)
        self.assertGreater(wide,narrow)

    def test_off_axis_approach_is_not_already_outside_a_gate_three_metres_ahead(self):
        gate=Gate(3,np.array([10.,0.,-4.]),np.array([1.,0.,0.,0.]))
        horizon,details=safe_horizon([7.,.7,-4.],[5.,0.,0.],[gate],1.7)
        self.assertAlmostEqual(horizon,.150)
        self.assertLess(details[0]['margin_m'][0],0.)
        self.assertGreater(details[0]['longitudinal_entry_s'],.15)

    def test_frame_budget_shrinks_when_approach_enters_the_inflated_gate_slab(self):
        gate=Gate(3,np.array([10.,0.,-4.]),np.array([1.,0.,0.,0.]))
        near,_=safe_horizon([9.3,.7,-4.],[5.,0.,0.],[gate],1.7)
        at_frame,_=safe_horizon([10.,.7,-4.],[5.,0.,0.],[gate],1.7)
        self.assertAlmostEqual(near,.020)
        self.assertEqual(at_frame,0.)

    def test_partial_mission_is_not_rewarded_with_large_J(self):
        trace=[{"t":i*.02,"state_cost":2.} for i in range(151)]
        self.assertAlmostEqual(paper_score(trace,3.)["J"],1/42)
        self.assertIsNone(paper_score(trace[:-1],3.))
        self.assertEqual(len(scenarios(12)),16)
        self.assertEqual({s["critical_index"] for s in scenarios(12)}, {0,11})
        self.assertEqual(len(config(12)["models"]),12)
        self.assertIsNone(score({"mission_passed":False,"clock_valid":True,"expired_not_applied":0,
            "rpc_after_deadline":0,"total_J":100.}))

    def test_expected_gates_are_derived_from_reference_not_observed_flight(self):
        from controller_architecture_study.safety_comparison_flight import reference_crossings
        gates = [Gate(i,np.array([x,0.,-4.]),np.array([1.,0.,0.,0.])) for i,x in ((1,10.),(2,20.),(3,30.))]
        self.assertEqual(reference_crossings([np.array([9.,0.,-4.]),np.array([21.,0.,-4.])],gates),[1,2])
        self.assertEqual(reference_crossings([np.array([11.,0.,-4.]),np.array([29.,0.,-4.])],gates),[2])

    def test_deadline_miss_or_abort_is_not_a_clean_safety_win(self):
        baseline = {"collisions":1, "mission_passed":False, "clock_valid":True,
            "expired_not_applied":0, "rpc_after_deadline":0, "error":None}
        framework = {**baseline, "collisions":0, "mission_passed":True}
        self.assertTrue(clean_split({"static":baseline, "framework":framework}))
        for key, value in (("expired_not_applied",1), ("rpc_after_deadline",1), ("clock_valid",False)):
            self.assertFalse(clean_split({"static":{**baseline,key:value}, "framework":framework}))
        self.assertFalse(clean_split({"static":baseline, "framework":{**framework,"mission_passed":False}}))


if __name__ == "__main__":
    unittest.main()
