from copy import deepcopy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import threading
import time

import numpy as np

from airsim_interface import RawDroneState
from controller_architecture_study.framework_runtime import (
    analyze_budget, bounded_plan, ModeMap, simulate_pipeline, state_bin, account_planning_delay, remaining_stage_cost)
from controller_architecture_study.framework_runtime_io import ProcessCommandIO
from controller_architecture_study.framework_runtime_worker import RuntimeServer
from controller_architecture_study.framework_runtime_world import FrameworkWorld
from controller_architecture_study.run_framework_runtime import clean_win, frozen_baseline, repeated_static_selection
from controller_architecture_study.framework_runtime_editor import parse_flag
from controller_architecture_study.framework_runtime_costs import stage_measurements
from concurrent.futures import Future


def row(name, deadline, release=0., model="small", stage="input"):
    return {"sample_id":name,"vehicle":name,"deadline_s":deadline,"planned_release_s":release,
        "model":model,"priority":0,"stage":stage}


class FrameworkRuntimeTests(unittest.TestCase):
    def test_static_screening_does_not_retry_excluded_or_accept_partially_failed_candidates(self):
        candidates = [{"name":name,"score":score,"configuration":{"name":name},"report":{"accepted":True}}
            for name,score in (("already_failed",20.),("fails_repeat",10.),("stable",9.))]
        calls = []
        def check(i,j,config):
            calls.append((config["name"],j))
            ok = not (config["name"]=="fails_repeat" and j==1)
            return {"accepted":ok,"total_J":9.5 if ok else None}
        with patch("controller_architecture_study.run_framework_runtime.atomic_json"):
            best = repeated_static_selection(candidates,check,None,excluded=("already_failed",))
        self.assertEqual(best["name"],"stable")
        self.assertAlmostEqual(best["score"],(9.+9.5+9.5)/3)
        self.assertEqual(len(calls),4)
        self.assertNotIn("already_failed",[name for name,_ in calls])
        self.assertEqual(candidates[1]["score"],10.)

    def test_safety_win_requires_matching_initial_states_and_valid_baseline_timing(self):
        pair = {"static":{"collisions":1,"timing_passed":True,"clock_valid":True,"error":None},
            "adaptive":{"accepted":True}}
        self.assertFalse(clean_win(pair))
        pair["initial_state_match"] = True
        self.assertTrue(clean_win(pair))
        pair["static"]["timing_passed"] = False
        self.assertFalse(clean_win(pair))

    def test_cost_visibility_does_not_include_post_abort_cleanup_receipts(self):
        rows = [{"sample_submitted_s":1.,"sample_response_received_s":1.003,"feature_done_s":1.004},
            {"sample_submitted_s":1.002,"sample_response_received_s":10.}]
        values,cutoff = stage_measurements(rows,[],{"sample_io_s":("sample_response_received_s","sample_submitted_s")})
        self.assertAlmostEqual(cutoff,1.004)
        self.assertEqual(len(values["sample_io_s"]),1)
        self.assertAlmostEqual(values["sample_io_s"][0],.003)

    def test_editor_query_is_parsed_as_data_not_assumed_disabled(self):
        self.assertIsNone(parse_flag("Unrecognized property"))
        self.assertTrue(parse_flag("[2026.10.24][1]True\n"))
        self.assertFalse(parse_flag("[2026.10.24][1]False\n"))

    def test_reusing_search_never_reuses_a_failed_or_different_domain_baseline(self):
        identity = {k:1 for k in ("vehicles","net_width_m","provenance","mode_map_sha256",
            "physical_phase_span_s","phase_s","route_sha256","duration_s","cpu_placement")}
        selected = {"candidate":{"configuration":{"models":{"a":"small"}},"report":{"accepted":True}}}
        cfg = frozen_baseline(selected,identity,identity)
        cfg["models"]["a"] = "large"
        self.assertEqual(selected["candidate"]["configuration"]["models"]["a"],"small")
        # The adaptive map does not affect the frozen static configuration.
        self.assertEqual(frozen_baseline(selected,identity,{**identity,"mode_map_sha256":2}),
            selected["candidate"]["configuration"])
        with self.assertRaises(ValueError):
            frozen_baseline(selected,identity,{**identity,"vehicles":2})
        with self.assertRaises(ValueError):
            frozen_baseline(selected,identity,{**identity,"editor_background_throttle_enabled":False})
        selected["candidate"]["report"]["accepted"] = False
        with self.assertRaises(ValueError):
            frozen_baseline(selected,identity,identity)
    def test_sparse_tail_observation_is_not_misreported_as_observed_state_error(self):
        world = FrameworkWorld.__new__(FrameworkWorld)
        world.traces = {"a":[{"t":t,"sim_ns":int(t*1e9),"sample_s":100+t,"position":[t,0.,-2.],
            "reference_position":[t,0.,-2.],"position_error_m":.01,"velocity_error_mps":.01,
            "risk":{"h":.4},"state_cost":.01} for t in (0.,2.7,2.82,3.)]}
        world.failures,world.gates,world.expected,world.net_width = {"a":None},{"a":[]},{"a":[]},2.
        report = world.physical_report(3.)["a"]
        self.assertTrue(report["physical_task_passed"])
        self.assertTrue(report["tail_state_bounds_observed_passed"])
        self.assertFalse(report["tail_observation_coverage_passed"])
        self.assertFalse(report["tail_tracking_passed"])

    def test_nominal_overrun_uses_frozen_tail_without_restarting_elapsed_time(self):
        costs = {"command_s":.003,"tail_costs":{"command_s":.017}}
        self.assertEqual(remaining_stage_cost(costs,"command_s",.002),(.001,False))
        remaining,tail = remaining_stage_cost(costs,"command_s",.004)
        self.assertTrue(tail)
        self.assertAlmostEqual(remaining,.013)
        self.assertLess(remaining,.017)
        self.assertLess(remaining_stage_cost(costs,"command_s",.020)[0],0.)

    def test_unknown_ingress_with_exhausted_tail_is_not_silently_zero_cost(self):
        job = {**row("a",.080,stage="compute"),"ingress_remaining_s":-.001}
        plan,info = bounded_plan(.020,{},[job],self.profile,self.costs)
        self.assertIsNone(plan)
        self.assertEqual(info["invalid_cost_contexts"][0]["ingress_remaining_s"],-.001)

    def test_executor_ack_makes_arrival_known_without_completing_or_changing_job(self):
        server = RuntimeServer.__new__(RuntimeServer)
        original = row("a",.1)
        server.rows = {"a":original}
        server._receive(("accepted","a","small",.1,.02))
        self.assertEqual(original["worker_enqueue_s"],.02)
        self.assertEqual(original["deadline_s"],.1)
        self.assertNotIn("ready_s",original)
        self.assertIn("worker_accept_received_s",original)
        with self.assertRaises(RuntimeError):
            server._receive(("accepted","a","small",.2,.03))

    def test_communication_ack_is_visible_before_post_command_state_finishes(self):
        io = ProcessCommandIO.__new__(ProcessCommandIO)
        original = row("a",.1)
        io.rows = {"a":original}
        io.futures = {"a":Future()}
        io.samples = {}
        io._receive(("update",{**original,"rpc_return_s":.04,"status":"applied"}))
        self.assertEqual(original["rpc_return_s"],.04)
        self.assertIn("command_response_received_s",original)
        self.assertFalse(io.futures["a"].done())
        io._receive(("command_done","state",{**original,"io_finished_s":.07}))
        self.assertEqual(io.futures["a"].result(),("state",original))
        self.assertIn("io_response_received_s",original)

    def test_communication_helper_cannot_move_original_deadline(self):
        io = ProcessCommandIO.__new__(ProcessCommandIO)
        original = row("a",.1)
        io.rows = {"a":original}
        with self.assertRaises(RuntimeError):
            io._receive(("update",{**original,"deadline_s":.2}))
        self.assertEqual(original["deadline_s"],.1)

    def setUp(self):
        self.profile = {"small":{"segment_estimates_s":[.001,.001]},
            "large":{"segment_estimates_s":[.004]*4}}
        self.costs = {"input_s":.002,"output_s":.002,"dispatch_guard_s":.002}

    def test_pipeline_overlaps_frontend_with_backend_but_serializes_each_resource(self):
        jobs = [row("a",.030),row("b",.030)]
        passed,finish,_ = simulate_pipeline(0.,jobs,[],self.profile,self.costs)
        self.assertTrue(passed)
        self.assertLess(max(finish.values()),.012)
        self.assertEqual(len(finish),2)

    def test_one_inflight_cap_is_charged(self):
        jobs = [row("a",.030),row("b",.030)]
        _,parallel,_ = simulate_pipeline(0.,jobs,[],self.profile,self.costs,inflight_limit=3)
        _,serial,_ = simulate_pipeline(0.,jobs,[],self.profile,self.costs,inflight_limit=1)
        self.assertLess(max(parallel.values()),max(serial.values()))

    def test_input_stage_does_not_block_an_imminent_urgent_output(self):
        profile = {"small":{"segment_estimates_s":[.00025]*5}}
        costs = {"input_s":.0045,"output_s":.0035,"dispatch_guard_s":.002}
        jobs = [row("urgent",.0108),row("later1",.075),row("later2",.080)]
        passed,finish,_ = simulate_pipeline(0.,jobs,[],profile,costs)
        self.assertTrue(passed)
        self.assertLessEqual(finish["urgent"],.0108)

    def test_pending_job_remaining_cost_and_output_are_not_ignored(self):
        pending = [row("running",.019,model="large",stage="compute")]
        passed,_,_ = simulate_pipeline(0.,[],pending,self.profile,self.costs)
        self.assertFalse(passed)  # full16ms + one4ms segment blocking +2ms output
        pending[0]["deadline_s"] = .030
        self.assertTrue(simulate_pipeline(0.,[],pending,self.profile,self.costs)[0])

    def test_bounded_plan_preserves_deadlines_modes_and_existing_jobs(self):
        future = {"a":row("a",.030),"b":row("b",.040)}
        original = deepcopy(future)
        plan,info = bounded_plan(0.,future,[],self.profile,self.costs)
        self.assertTrue(info["feasible"])
        self.assertLessEqual(info["checked"],5)
        self.assertEqual(future,original)
        for v in future:
            self.assertEqual(plan[v]["deadline_s"],future[v]["deadline_s"])
            self.assertEqual(plan[v]["model"],future[v]["model"])
        self.assertTrue(simulate_pipeline(0.,list(plan.values()),[],self.profile,self.costs)[0])

    def test_bounded_plan_can_desynchronize_future_captures_without_moving_deadlines(self):
        future = {"a":row("a",.100),"b":row("b",.100),"c":row("c",.100)}
        plan,info = bounded_plan(0.,future,[],self.profile,self.costs)
        self.assertTrue(info["feasible"])
        self.assertEqual(info["release_pattern"],"staggered")
        self.assertLess(plan["a"]["planned_release_s"],plan["b"]["planned_release_s"])
        self.assertLess(plan["b"]["planned_release_s"],plan["c"]["planned_release_s"])
        self.assertTrue(all(r["deadline_s"]==.100 for r in plan.values()))
        self.assertLessEqual(info["checked"],5)

    def test_no_infeasible_plan_falls_through_to_dispatch(self):
        plan,info = bounded_plan(0.,{"a":row("a",.003)},[],self.profile,self.costs)
        self.assertIsNone(plan)
        self.assertFalse(info["feasible"])
        self.assertLessEqual(info["checked"],5)

    def test_release_search_does_not_sum_parallel_ipc_as_serial_frontend_work(self):
        costs = {"input_s":.001,"output_s":.001,"sample_io_s":.001,"io_s":.003,
            "command_s":.001,"handoff_s":.0002,"ingress_s":.020,"egress_s":.020,
            "dispatch_guard_s":.002,"completion_guard_s":.001}
        future = {str(i):row(str(i),.080) for i in range(7)}
        plan,info = bounded_plan(0.,future,[],self.profile,costs,inflight_limit=7)
        self.assertIsNotNone(plan)
        self.assertGreater(info["release_base_s"],.010)
        self.assertLessEqual(info["checked"],5)
        self.assertTrue(simulate_pipeline(0.,list(plan.values()),[],self.profile,costs,inflight_limit=7)[0])

    def test_planner_reservation_blocks_frontend_not_ongoing_communication(self):
        costs = {**self.costs,"sample_io_s":.003,"io_s":.002,"command_s":.001,
            "handoff_s":.0005,"frontend_reserved_until_s":.006}
        pending = [{**row("a",.030,stage="sampling"),"sample_io_remaining_s":.001}]
        # At snapshot t=2ms the capture still has1ms. It completes during the
        # planner's reservation; do NOT move t to6ms and invent negative residual.
        passed,finish,_ = simulate_pipeline(.002,[],pending,self.profile,costs)
        self.assertTrue(passed)
        self.assertGreaterEqual(finish["a"],.011)
        self.assertLess(finish["a"],.015)
        plan,info = bounded_plan(.002,{"b":row("b",.040)},pending,self.profile,costs)
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(plan["b"]["planned_release_s"],.006)
        self.assertLessEqual(info["checked"],5)

    def test_no_new_capture_is_launched_while_frontend_is_reserved(self):
        costs = {**self.costs,"sample_io_s":.003,"io_s":.002,"command_s":.001,
            "handoff_s":.0005,"frontend_reserved_until_s":.006}
        passed,finish,_ = simulate_pipeline(.002,[row("a",.040,release=.002)],[],self.profile,costs)
        self.assertTrue(passed)
        self.assertGreaterEqual(finish["a"],.014)

    def test_exhausted_sampling_estimate_is_identified_as_a_cost_model_problem(self):
        pending = [{**row("a",.100,stage="sampling"),"sample_io_remaining_s":-.001}]
        plan,info = bounded_plan(.010,{},pending,self.profile,self.costs)
        self.assertIsNone(plan)
        self.assertEqual(info["checked"],0)
        self.assertEqual(info["invalid_cost_contexts"][0]["sample_id"],"a")
        self.assertIn("empirical cost model",info["reason"])

    def test_planner_lag_must_be_paid_by_original_slack_not_a_deadline_extension(self):
        proposal = {"a":row("a",.030,release=.010)}
        info = {"finish":{"a":.020,"b":.009}}
        pending = [{**row("b",.010,stage="analysis"),"command_completed_s":.009}]
        shifted = account_planning_delay(proposal,info,pending,.005,.008,self.costs)
        self.assertAlmostEqual(shifted["a"]["planned_release_s"],.013)
        self.assertEqual(shifted["a"]["deadline_s"],.030)
        self.assertEqual(proposal["a"]["planned_release_s"],.010)
        self.assertEqual(info["finish"]["b"],.009)
        self.assertAlmostEqual(info["finish"]["a"],.023)
        self.assertIsNone(account_planning_delay(proposal,{"finish":{"a":.029}},[],.005,.008,self.costs))

    def test_unknown_region_does_not_implicitly_select_the_largest_model(self):
        model_map = ModeMap.__new__(ModeMap)
        model_map.data = {"bins":{},"analysis_to_next_command_limit_s":.080}
        selected,info = model_map.select({"h":.4,"hdot":0.},self.profile)
        self.assertIsNone(selected)
        self.assertEqual(info["status"],"unknown_or_no_admissible_mode")
        self.assertIsNone(state_bin({"h":-.01,"hdot":0.}))
        self.assertIsNone(state_bin({"h":.3,"hdot":3.}))

    def test_selects_cheapest_among_only_admissible_modes_without_monotonicity_assumption(self):
        model_map = ModeMap.__new__(ModeMap)
        model_map.data = {"bins":{"h1_v2":{"modes":{
            "small":{"status":"unknown"},"large":{"status":"empirical_admissible"}}}},
            "analysis_to_next_command_limit_s":.080}
        selected,_ = model_map.select({"h":.4,"hdot":0.},self.profile)
        self.assertEqual(selected,"large")
        model_map.data["bins"]["h1_v2"]["modes"]["small"]["status"] = "empirical_admissible"
        self.assertEqual(model_map.select({"h":.4,"hdot":0.},self.profile)[0],"small")

    def test_mode_specific_wait_limit_is_not_extended_to_the_global_update_cap(self):
        model_map = ModeMap.__new__(ModeMap)
        cell = {"status":"empirical_admissible","delay_limit_s":.020}
        model_map.data = {"bins":{"h0_v1":{"modes":{"small":cell}}},
            "analysis_to_next_command_limit_s":.080}
        _,selection = model_map.select({"h":.15,"hdot":-.5},self.profile)
        self.assertEqual(selection["delay_limit_s"],.020)
        for invalid in (0.,.081,float("nan")):
            cell["delay_limit_s"] = invalid
            with self.assertRaises(ValueError):
                model_map.select({"h":.15,"hdot":-.5},self.profile)

    def test_local_refinement_does_not_promote_the_entire_failed_parent_bin(self):
        model_map = ModeMap.__new__(ModeMap)
        model_map.data = {"bins":{},"analysis_to_next_command_limit_s":.080,"local_cells":[{
            "id":"local","h_range":[.225,.25],"hdot_range":[-.35,-.2],
            "modes":{"small":{"status":"empirical_admissible"}}}]}
        self.assertEqual(model_map.select({"h":.24,"hdot":-.3},self.profile)[0],"small")
        self.assertIsNone(model_map.select({"h":.15,"hdot":-.5},self.profile)[0])

    def test_deadline_uses_a_fresh_post_command_observation_and_original_update_cap(self):
        state = RawDroneState(np.zeros(3),np.array([1.,0.,0.,0.]),np.zeros(3),np.zeros(3),np.zeros(3))
        with patch("controller_architecture_study.framework_runtime.safe_horizon",return_value=(.050,[])):
            budget = analyze_budget(state,[9.8,0.,0.,0.],10.020,10.,[],2.)
        self.assertAlmostEqual(budget["deadline_s"],10.070)
        self.assertEqual(budget["held_control"],[9.8,0.,0.,0.])
        with self.assertRaises(ValueError):
            analyze_budget(state,[9.8,0.,0.,0.],9.99,10.,[],2.)

    def test_duplicate_job_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_pipeline(0.,[row("a",.1)],[row("a",.1)],self.profile,self.costs)

    def test_parallel_command_io_is_bounded_and_included_in_response_time(self):
        costs = {"input_s":.001,"output_s":.001,"handoff_s":.0001,"io_s":.020,
            "io_slots":2,"dispatch_guard_s":.002}
        jobs = [row("a",.030),row("b",.030)]
        good,finish,_ = simulate_pipeline(0.,jobs,[],self.profile,costs)
        self.assertTrue(good)
        self.assertGreater(min(finish.values()),.020)
        costs["io_slots"] = 1
        self.assertFalse(simulate_pipeline(0.,jobs,[],self.profile,costs)[0])

    def test_pending_io_uses_only_remaining_io_not_the_model_compute_again(self):
        costs = {**self.costs,"io_s":.030,"io_slots":2}
        pending = [{**row("a",.015,model="large",stage="io"),"io_remaining_s":.010}]
        good,finish,_ = simulate_pipeline(0.,[],pending,self.profile,costs)
        self.assertTrue(good)
        self.assertAlmostEqual(finish["a"],.010)

    def test_pipeline_charges_ipc_both_before_and_after_compute(self):
        costs = {**self.costs,"ingress_s":.003,"egress_s":.004}
        passed,finish,_ = simulate_pipeline(0.,[row("a",.020)],[],self.profile,costs)
        self.assertTrue(passed)
        self.assertAlmostEqual(finish["a"],.013)
        self.assertFalse(simulate_pipeline(0.,[row("a",.012)],[],self.profile,costs)[0])

    def test_state_rpc_is_parallel_io_but_feature_and_inference_resources_are_shared(self):
        costs = {"input_s":.001,"sample_io_s":.010,"output_s":.001,"handoff_s":.001,
            "io_s":.040,"io_slots":8,"dispatch_guard_s":.002}
        jobs = [row(str(i),.080) for i in range(8)]
        passed,finish,_ = simulate_pipeline(0.,jobs,[],self.profile,costs,inflight_limit=8)
        self.assertTrue(passed)
        self.assertGreater(min(finish.values()),.050)
        self.assertEqual(len(finish),8)
        costs["io_slots"] = 1
        self.assertFalse(simulate_pipeline(0.,jobs,[],self.profile,costs,inflight_limit=8)[0])

    def test_pending_sample_is_not_charged_a_second_sampling_rpc(self):
        costs = {**self.costs,"sample_io_s":.010,"io_s":.020,"handoff_s":.001,"io_slots":3}
        pending = [{**row("a",.030,stage="sampling"),"sample_io_remaining_s":.002}]
        passed,finish,_ = simulate_pipeline(0.,[],pending,self.profile,costs)
        self.assertTrue(passed)
        self.assertAlmostEqual(finish["a"],.027)

    def test_next_job_analysis_cannot_starve_an_already_released_input(self):
        costs = {"input_s":.001,"sample_io_s":.010,"output_s":.005,
            "handoff_s":.001,"io_s":.020,"io_slots":3}
        pending = [row("urgent",.026,stage="input_ready"),
            {**row("acknowledged",.005,stage="analysis"),"command_completed_s":0.}]
        passed,finish,_ = simulate_pipeline(0.,[],pending,self.profile,costs)
        self.assertTrue(passed)
        self.assertLessEqual(finish["urgent"],.026)

    def test_direct_features_preserve_original_237d_values(self):
        from controller_architecture_study.fast_convex_preprocess import FastReferenceWindow, encode
        from controller_architecture_study.framework_runtime_features import encode_direct, reference_point
        from controller_architecture_study.state_mode_calibration import moderate_case
        from controller_architecture_study.fly_decision_dnn import StoredRoutePlanner
        _,layout = moderate_case()
        planner = StoredRoutePlanner(layout)
        trajectory = planner.trajectory
        window = FastReferenceWindow(trajectory,planner.config)
        rng = np.random.default_rng(42)
        for elapsed in np.linspace(0.,trajectory.duration+1.,20):
            refs = window.window(elapsed)
            ref = refs[0]
            state = RawDroneState(ref.position+rng.normal(0.,.3,3),
                ref.orientation,ref.velocity+rng.normal(0.,.5,3),np.zeros(3),np.zeros(3))
            np.testing.assert_array_equal(encode_direct(window,state,elapsed),encode(state,refs))
            np.testing.assert_allclose(reference_point(window,elapsed).orientation,ref.orientation,atol=2e-15,rtol=0.)

    def test_acknowledged_command_is_not_late_just_because_fresh_state_arrives_later(self):
        costs = {**self.costs,"io_s":.030,"io_slots":2}
        pending = [{**row("a",.010,stage="io"),"io_remaining_s":.005,"command_completed_s":.009}]
        good,finish,_ = simulate_pipeline(.015,[],pending,self.profile,costs)
        self.assertTrue(good)
        self.assertEqual(finish["a"],.009)

    def test_command_deadline_is_not_the_end_of_the_following_state_rpc(self):
        costs = {"input_s":.001,"output_s":.001,"handoff_s":.001,"io_s":.040,
            "command_s":.020,"io_slots":2}
        passed,finish,_ = simulate_pipeline(0.,[row("a",.026)],[],self.profile,costs)
        self.assertTrue(passed)
        self.assertAlmostEqual(finish["a"],.024)
        # The whole40ms I/O operation still occupies the shared channel.
        costs["io_slots"] = 1
        self.assertFalse(simulate_pipeline(0.,[row("a",.026),row("b",.040)],[],self.profile,costs)[0])

    def test_completion_margin_also_protects_against_lower_priority_cpu_blocking(self):
        profile = {"small":{"segment_estimates_s":[.0001595,.0001865,.00018875,.00019925,.0001553]}}
        costs = {"input_s":.003,"sample_io_s":.020,"output_s":.0015,"handoff_s":.001,
            "io_s":.040,"command_s":.025,"ingress_s":.0015,"egress_s":.0015,
            "completion_guard_s":.001,"io_slots":3}
        jobs = [row("urgent",.0555979),row("later1",.0633404),row("later2",.0647787)]
        passed,finish,_ = simulate_pipeline(0.,jobs,[],profile,costs)
        self.assertTrue(passed)
        self.assertLessEqual(finish["urgent"]+.001,jobs[0]["deadline_s"])

    def test_command_io_owns_one_client_per_thread_and_preserves_deadlines(self):
        from controller_architecture_study.framework_runtime_io import CommandIO
        from config import load_calibration
        class Client:
            def __init__(self, **kwargs):
                self.owner = threading.get_ident()
                self.client = SimpleNamespace(close=lambda:None)
            def ping(self):
                assert self.owner == threading.get_ident()
                return True
            def getMultirotorState(self,**kwargs):
                assert self.owner == threading.get_ident()
            def moveByAngleRatesThrottleAsync(self,*args,**kwargs):
                assert self.owner == threading.get_ident()
        with patch("controller_architecture_study.framework_runtime_io.require_airsim",return_value=SimpleNamespace(MultirotorClient=Client)), \
                patch("controller_architecture_study.framework_runtime_io.FrameworkWorld.snapshot_from",return_value=(None,None,time.perf_counter())):
            channel = CommandIO(2,load_calibration(),.020)
            try:
                r = row("a",time.perf_counter()+.100)
                channel.send(np.array([9.8,0.,0.,0.]),r)
                values = []
                until = time.perf_counter()+1.
                while not values and time.perf_counter()<until:
                    values = channel.drain()
                    time.sleep(.001)
                self.assertEqual(r["status"],"applied")
                self.assertLess(r["rpc_return_s"],r["deadline_s"])
                self.assertEqual(len(channel.clients),2)
            finally:
                channel.close()


if __name__ == "__main__":
    unittest.main()
