#!/usr/bin/env python3
"""Synthetic/stdlib-only analysis tests; never connects to a board or AirSim."""
import json
import math
import tempfile
import unittest
from pathlib import Path
import analyze_flight as a


class AnalysisTests(unittest.TestCase):
    def test_quantiles(self):
        self.assertEqual(a.percentile([0,10],.95),9.5)
        self.assertEqual(a.stats([])["p99"],None)

    def test_native_gate_forward_rule(self):
        gate=dict(index=1,center=[0,0,0],orientation=[1,0,0,0])
        p=dict(position=[-1,0,0],t=0)
        q=dict(position=[1,0,0],t=.05)
        cross=a.segment_crossing(p,q,gate)
        self.assertAlmostEqual(cross["time_s"],.025)
        self.assertIsNone(a.segment_crossing(q,p,gate))
        self.assertIsNone(a.segment_crossing(dict(position=[0,0,0],t=0),dict(position=[0,0,0],t=1),gate))

    def test_rotated_gate(self):
        # Gate normal +Y in world. Local lateral +1 maps to world -X.
        gate=dict(index=1,center=[1,2,3],orientation=[math.sqrt(.5),0,0,math.sqrt(.5)])
        self.assertAlmostEqual(a.gate_local([0,2,3],gate)[1],1)
        c=a.segment_crossing(dict(position=[1,1,3],t=0),dict(position=[1,3,3],t=1),gate)
        self.assertAlmostEqual(c["time_s"],.5)

    def test_gate_body_is_stricter(self):
        gate=dict(index=1,center=[0,0,0],orientation=[1,0,0,0])
        trace=[dict(position=[-1,1.5,0],t=0),dict(position=[1,1.5,0],t=.05)]
        r=a.gate_report(trace,dict(gates=[gate]),{})
        self.assertTrue(r["all_point_crossings_observed"])
        self.assertFalse(r["all_body_crossings_observed"])
        trace[1]["t"]=1.
        self.assertFalse(a.gate_report(trace,dict(gates=[gate]),{})["all_crossing_gaps_acceptable"])

    def test_reference_clamp(self):
        l=dict(segments=[dict(start_time=0,duration=2,coefficients=[[0,1,0,0,0,0],[1,0,0,0,0,0],[2,0,0,0,0,0]])])
        self.assertEqual(a.reference(l,1),([1,1,2],[1,0,0]))
        self.assertEqual(a.reference(l,9)[0],[2,1,2])

    def test_timing_missing_and_negative(self):
        errors=[]
        result,rows=a.phase_values([dict(dpu_submit_us=100,dpu_result_readable_us=90),dict(dpu_submit_us=100,dpu_result_readable_us=200)],errors)
        self.assertEqual(result["dpu_call"]["count"],1)
        self.assertAlmostEqual(result["dpu_call"]["p50"],.1)
        self.assertEqual(len(errors),1)
        self.assertNotIn("dpu_call_ms",rows[0])

    def test_collision_does_not_become_success(self):
        rows=[dict(status="delivered",error="collision_after_command",command_sent=True),
              dict(status="input_range_rejected",dpu_status=3),
              dict(status="error",command_may_have_been_sent=True,command_sent=False)]
        c=a.request_counts(rows)
        self.assertEqual(c["delivered_status"],1)
        self.assertEqual(c["delivered_without_recorded_error"],0)
        self.assertEqual(c["collision_events"],1)
        self.assertEqual(c["input_range_rejected"],1)
        self.assertEqual(c["command_delivery_unknown"],1)

    def test_partial_failure_and_no_full_window_extrapolation(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/"execution_identity.json").write_text(json.dumps(dict(origin_us=1000000,duration_s=1,period_ms=50,reference_start_s=0)))
            layout=dict(segments=[dict(start_time=0,duration=1,coefficients=[[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])],gates=[])
            (p/"prepared_source.json").write_text(json.dumps(dict(layouts=dict(Drone1=layout))))
            rows=[dict(vehicle="Drone1",cycle=i,status="delivered",sample_start_us=1000000+i*50000,
                       position=[i*.05,0,0],velocity=[1,0,0],sim_timestamp_ns=1000000000+i*50000000) for i in range(3)]
            (p/"events.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            (p/"failure.json").write_text(json.dumps(dict(error="inference_status")))
            result,_,_=a.analyze_run(p)
            self.assertTrue(result["audit_passed"])
            self.assertTrue(result["used_partial_journal"])
            self.assertFalse(result["complete_flight_observed"])
            t=result["vehicles"]["Drone1"]["tracking"]
            self.assertFalse(t["window_covered"])
            self.assertIsNone(t["full_window_tracking_rmse_m"])
            self.assertEqual(t["observed_sample_rmse_m"],0)
            self.assertAlmostEqual(t["sim_wall_ratio"],1)

    def test_bad_journal_and_duplicates_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"events.jsonl"
            p.write_text('{"vehicle":"A","cycle":0}\n{"vehicle":"A","cycle":0}\n{broken\n')
            errors=[]
            rows=a.load_rows(p,errors)
            self.assertEqual(len(rows),1)
            self.assertEqual(len(errors),2)

    def test_boundary_proxies_and_final_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            identity=dict(origin_us=1000000,duration_s=.1,period_ms=50,reference_start_s=0)
            layout=dict(segments=[dict(start_time=0,duration=1,coefficients=[[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])],gates=[])
            (p/"execution_identity.json").write_text(json.dumps(identity))
            (p/"prepared_source.json").write_text(json.dumps(dict(layouts=dict(A=layout))))
            rows=[dict(vehicle="A",cycle=i,status="delivered",sample_start_us=1001000+i*50000,
                position=[.001+i*.05,0,0],velocity=[1,0,0],sim_timestamp_ns=1001000000+i*50000000) for i in range(2)]
            (p/"requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            initial=[dict(vehicle="A",position=[0,0,0],velocity=[1,0,0],sim_timestamp_ns=1000000000,observed_us=900000)]
            final=[dict(vehicle="A",position=[.101,0,0],velocity=[1,0,0],sim_timestamp_ns=1101000000,observed_us=1101000,collision_new=True)]
            (p/"initial_states.json").write_text(json.dumps(initial))
            (p/"final_states.json").write_text(json.dumps(final))
            result,_,_=a.analyze_run(p)
            vehicle=result["vehicles"]["A"]
            self.assertTrue(vehicle["tracking"]["window_covered"])
            self.assertAlmostEqual(vehicle["tracking"]["full_window_tracking_mse_m2"],0)
            self.assertEqual(len(vehicle["tracking"]["boundary_warnings"]),2)
            self.assertEqual(vehicle["final_snapshot_new_collisions"],1)
            self.assertFalse(vehicle["flight_passed_observed"])

    def test_end_to_end_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); data=p/"run";data.mkdir();out=p/"analysis"
            (data/"events.jsonl").write_text('{"vehicle":"A","cycle":0,"status":"input_range_rejected","dpu_status":3}\n')
            self.assertEqual(a.main(["--data",str(data),"--output",str(out)]),0)
            self.assertTrue((out/"REPORT.md").is_file())
            self.assertTrue((out/"timing_per_request.csv").is_file())
            with self.assertRaises(FileExistsError):a.main(["--data",str(data),"--output",str(out)])

    def test_paused_readback_delay_is_not_active_simulation_slowdown(self):
        identity=dict(origin_us=1000000,duration_s=.1,reference_start_s=0)
        layout=dict(segments=[dict(start_time=0,duration=2,coefficients=[[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])],gates=[])
        rows=[dict(vehicle='A',cycle=i,sample_start_us=1001000+i*50000,
                   position=[.001+i*.05,0,0],velocity=[1,0,0],sim_timestamp_ns=1001000000+i*50000000) for i in range(2)]
        initial=[dict(vehicle='A',position=[0,0,0],velocity=[1,0,0],sim_timestamp_ns=1000000000,observed_us=900000)]
        final=[dict(vehicle='A',position=[.101,0,0],velocity=[1,0,0],sim_timestamp_ns=1101000000,
                    observed_us=2101000,pause_return_us=1101000)]
        errors=[]
        report,trace=a.tracking(rows,identity,dict(layouts=dict(A=layout)),'A',errors,initial,final)
        self.assertEqual(errors,[])
        self.assertTrue(report['sim_clock_observed_valid'])
        self.assertAlmostEqual(report['sim_wall_ratio'],1.)
        self.assertAlmostEqual(report['last_sample_s'],.101)
        self.assertAlmostEqual(report['full_window_tracking_mse_m2'],0.)
        self.assertEqual(trace[-1]['ps_timestamp_scope'],'pause_return_upper_bound_proxy')
        self.assertEqual(trace[-1]['observed_us'],2101000)

    def test_failed_empty_final_export_uses_actual_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            journal='{"vehicle":"A","cycle":0,"status":"delivered"}\n'
            (p/"events.jsonl").write_text(journal)
            (p/"requests.jsonl").write_text("")
            (p/"failure.json").write_text('{"error":"failed final summary"}')
            result,_,_=a.analyze_run(p)
            self.assertTrue(result["audit_passed"])
            self.assertTrue(result["partial_final_export"])
            self.assertTrue(result["used_partial_journal"])
            self.assertEqual(result["counts"]["recorded_slots"],1)
            self.assertEqual(result["final_export_rows"],0)
            self.assertEqual(result["journal_rows"],1)
            self.assertEqual(len(result["export_warnings"]),1)
            self.assertIn(str(p/"requests.jsonl"),result["sha256"])
            self.assertFalse(result["complete_flight_observed"])
            self.assertEqual((p/"events.jsonl").read_text(),journal)

    def test_unexplained_or_divergent_final_export_is_audit_error(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/"events.jsonl").write_text('{"vehicle":"A","cycle":0,"status":"delivered"}\n')
            (p/"requests.jsonl").write_text("")
            result,_,_=a.analyze_run(p)
            self.assertFalse(result["audit_passed"])
            self.assertFalse(result["partial_final_export"])
            (p/"failure.json").write_text('{"error":"failed final summary"}')
            (p/"requests.jsonl").write_text('{"vehicle":"A","cycle":0,"status":"error"}\n')
            result,_,_=a.analyze_run(p)
            self.assertFalse(result["audit_passed"])
            self.assertFalse(result["partial_final_export"])


if __name__=="__main__":
    unittest.main(verbosity=2)
