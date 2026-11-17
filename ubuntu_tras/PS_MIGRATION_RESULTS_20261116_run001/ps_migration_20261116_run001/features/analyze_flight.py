#!/usr/bin/env python3
"""Audit native PS flight logs without importing AirSim, NumPy, or teacher code.

All host-phase timestamps are PS CLOCK_MONOTONIC microseconds.  AirSim simulation
timestamps are used only for simulation-clock checks, never phase subtraction.
This is an observational report: a returned RPC is not physical actuation proof.
"""
from __future__ import print_function

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


# Name, start, end. Keep pure hardware time absent: DNNDK exposes call wall time.
PHASES = [
    ("release_lateness", "planned_release_us", "actual_publish_us"),
    ("publisher_to_sample", "actual_publish_us", "sample_start_us"),
    ("state_rpc", "sample_start_us", "state_ready_us"),
    ("collision_rpc", "state_ready_us", "collision_ready_us"),
    ("ps_lock_wait", "ps_lock_requested_us", "ps_lock_acquired_us"),
    ("reference_generation", "ps_lock_acquired_us", "reference_ready_us"),
    ("safety_budget", "reference_ready_us", "analysis_ready_us"),
    ("feature_construction", "analysis_ready_us", "feature_ready_us"),
    ("sample_to_feature_ready", "sample_start_us", "feature_ready_us"),
    ("dpu_queue", "dpu_enqueue_us", "dpu_dequeue_us"),
    ("input_preparation_quantization", "input_prep_start_us", "input_prep_end_us"),
    ("dpu_call", "dpu_submit_us", "dpu_result_readable_us"),
    ("dpu_output_processing", "dpu_result_readable_us", "dpu_output_ready_us"),
    ("local_dpu_service", "input_prep_start_us", "dpu_output_ready_us"),
    ("control_conversion", "output_ready_us", "control_converted_us"),
    ("command_rpc_proxy", "command_attempt_start_us", "command_return_us"),
    ("command_enqueue_to_proxy_return", "command_enqueued_us", "command_return_us"),
    ("paired_state_rpc", "paired_sample_start_us", "paired_state_ready_us"),
    ("paired_collision_rpc", "paired_state_ready_us", "paired_collision_ready_us"),
    ("paired_hold_analysis_with_lock_wait", "paired_collision_ready_us", "paired_analysis_ready_us"),
    ("sample_to_dpu_output", "sample_start_us", "output_ready_us"),
    ("sample_to_command_proxy", "sample_start_us", "command_return_us"),
    ("release_to_command_proxy", "planned_release_us", "command_return_us"),
    ("release_to_slot_return", "planned_release_us", "slot_return_us"),
]


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, fraction):
    """Linear interpolation at (n-1)*fraction, matching NumPy default."""
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * fraction
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def stats(values):
    values = [float(v) for v in values if finite_number(v)]
    return dict(count=len(values), min=min(values) if values else None,
                mean=sum(values) / len(values) if values else None,
                p50=percentile(values, .50), p95=percentile(values, .95),
                p99=percentile(values, .99), max=max(values) if values else None)


def vector(value, size):
    return isinstance(value, list) and len(value) == size and all(finite_number(v) for v in value)


def subtract(a, b):
    return [a[i] - b[i] for i in range(3)]


def norm(value):
    return math.sqrt(sum(v * v for v in value))


def rotate(q, v):
    """Normalized wxyz quaternion rotation, no frame/sign conversion."""
    length = math.sqrt(sum(c * c for c in q))
    if length < 1e-12:
        raise ValueError("zero gate quaternion")
    w, x, y, z = [c / length for c in q]
    return [
        (1-2*y*y-2*z*z)*v[0] + (2*x*y-2*z*w)*v[1] + (2*x*z+2*y*w)*v[2],
        (2*x*y+2*z*w)*v[0] + (1-2*x*x-2*z*z)*v[1] + (2*y*z-2*x*w)*v[2],
        (2*x*z-2*y*w)*v[0] + (2*y*z+2*x*w)*v[1] + (1-2*x*x-2*y*y)*v[2],
    ]


def gate_local(point, gate):
    q = gate["orientation"]
    if not vector(q, 4) or not vector(gate["center"], 3):
        raise ValueError("invalid gate geometry")
    return rotate([q[0], -q[1], -q[2], -q[3]], subtract(point, gate["center"]))


def segment_crossing(a, b, gate):
    """Original gate_route.py forward segment-plane intersection rule."""
    p0, p1 = gate_local(a["position"], gate), gate_local(b["position"], gate)
    if not p0[0] <= 0 < p1[0] or p1[0] - p0[0] <= 1e-12:
        return None
    fraction = -p0[0] / (p1[0] - p0[0])
    local = [p0[i] + fraction * (p1[i] - p0[i]) for i in range(3)]
    return dict(fraction=fraction, time_s=a["t"] + fraction * (b["t"]-a["t"]),
                lateral_offset_m=local[1], vertical_offset_m=local[2],
                observation_gap_ms=(b["t"]-a["t"]) * 1000,
                before_cycle=a.get("cycle"), after_cycle=b.get("cycle"))


def reference(layout, elapsed):
    segments = layout["segments"]
    if not segments or not finite_number(elapsed) or elapsed < 0:
        raise ValueError("invalid polynomial reference time/segments")
    segment = segments[-1]
    for candidate in segments:
        if elapsed < candidate["start_time"] + candidate["duration"]:
            segment = candidate
            break
    t = min(segment["duration"], max(0., elapsed-segment["start_time"]))
    coefficients = segment["coefficients"]
    if len(coefficients) != 3 or any(not vector(c, 6) for c in coefficients):
        raise ValueError("invalid reference polynomial coefficients")
    p = [sum(c[i] * t**i for i in range(6)) for c in coefficients]
    v = [sum(i*c[i] * t**(i-1) for i in range(1, 6)) for c in coefficients]
    return p, v


def load_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_rows(path, audit):
    rows, seen = [], set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("not a JSON object")
            key = (row.get("vehicle"), row.get("cycle"))
            if key in seen:
                raise ValueError("duplicate vehicle/cycle {}".format(key))
            seen.add(key)
            rows.append(row)
        except (ValueError, TypeError) as error:
            audit.append(dict(file=str(path), line=number, error=str(error)))
    return rows


def phase_values(rows, audit):
    values = collections.defaultdict(list)
    per_request = []
    for row in rows:
        item = dict(vehicle=row.get("vehicle"), cycle=row.get("cycle"),
                    model_id=row.get("model_id"), status=row.get("status"))
        for name, start, end in PHASES:
            a, b = row.get(start), row.get(end)
            if not finite_number(a) or not finite_number(b) or a <= 0 or b <= 0:
                continue
            if b < a:
                audit.append(dict(vehicle=row.get("vehicle"), cycle=row.get("cycle"),
                                  error="negative phase {}".format(name), start=a, end=b))
                continue
            milliseconds = (b-a) / 1000.
            values[name].append(milliseconds)
            item[name+"_ms"] = milliseconds
        per_request.append(item)
    return {name: stats(values[name]) for name, _, _ in PHASES}, per_request


def tracking(rows, identity, setup, name, audit, initial_states=None, final_states=None):
    layout = setup.get("layouts", {}).get(name)
    origin = identity.get("origin_us")
    reference_start = identity.get("reference_start_s", setup.get("reference_start_s", 1.5))
    trace = []
    if not finite_number(origin):
        return dict(status="NOT_MEASURED", reason="origin_us missing"), trace
    for row in rows:
        stamp = row.get("sample_start_us")
        if not finite_number(stamp) or not vector(row.get("position"), 3):
            continue
        elapsed = (stamp-origin) / 1e6
        if elapsed < 0:
            audit.append(dict(vehicle=name, cycle=row.get("cycle"), error="state before origin"))
            continue
        p, v = row.get("reference_position"), row.get("reference_velocity")
        scope = "recorded_reference"
        if not vector(p, 3) or not vector(v, 3):
            if layout is None:
                continue
            p, v = reference(layout, reference_start+elapsed)
            scope = "reconstructed_frozen_polynomial_at_recorded_sample_start"
        entry = dict(cycle=row.get("cycle"), t=elapsed, sim_timestamp_ns=row.get("sim_timestamp_ns"),
                     position=row["position"], reference_position=p, reference_velocity=v,
                     position_error_m=norm(subtract(row["position"], p)), reference_scope=scope,
                     observation_role="active_slot", ps_timestamp_scope="state_RPC_start_proxy")
        if vector(row.get("velocity"), 3):
            entry["velocity"] = row["velocity"]
            entry["velocity_error_mps"] = norm(subtract(row["velocity"], v))
        if finite_number(row.get("tracking_error_m")):
            entry["runtime_tracking_error_m"] = row["tracking_error_m"]
            entry["tracking_error_reconstruction_abs_m"] = abs(entry["position_error_m"]-row["tracking_error_m"])
        trace.append(entry)
    boundary_warnings=[]
    # These are additional actual observations, not invented positions. The
    # initial state is known while physics is paused. Final state is read after
    # pausing, therefore observed_us is explicitly only a boundary-time proxy.
    for role, states in [("initial_paused",initial_states or []),("final_paused",final_states or [])]:
        for row in states:
            if row.get("vehicle")!=name or not vector(row.get("position"),3) or layout is None:
                continue
            stamp=row.get("observed_us")
            if not finite_number(stamp):
                continue
            # A read performed long AFTER physics was paused is not evidence
            # of an equally long active simulation interval. Prefer the
            # recorded pause-return upper-bound proxy when available.
            pause_return=row.get("pause_return_us")
            pause_proxy=role=="final_paused" and finite_number(pause_return) and origin<=pause_return<=stamp
            elapsed=0. if role=="initial_paused" else ((pause_return if pause_proxy else stamp)-origin)/1e6
            if elapsed<0:
                continue
            if any(abs(existing["t"]-elapsed)<1e-12 for existing in trace):
                # An active state at the same instant is preferred to a proxy.
                continue
            p,v=reference(layout,reference_start+elapsed)
            entry=dict(cycle=None,t=elapsed,sim_timestamp_ns=row.get("sim_timestamp_ns"),
                position=row["position"],reference_position=p,reference_velocity=v,
                position_error_m=norm(subtract(row["position"],p)),
                reference_scope="reconstructed_frozen_polynomial_boundary_proxy",
                observation_role=role,ps_timestamp_scope="paused_initial_state_aligned_to_origin" if role=="initial_paused" else ("pause_return_upper_bound_proxy" if pause_proxy else "post_pause_readback_observed_us_proxy"),
                observed_us=stamp)
            if vector(row.get("velocity"),3):
                entry["velocity"]=row["velocity"]
                entry["velocity_error_mps"]=norm(subtract(row["velocity"],v))
            trace.append(entry)
            boundary_warnings.append("{} endpoint uses a paused-state timing proxy, not a hardware sampling instant".format(role))
    trace.sort(key=lambda r:r["t"])
    if not trace:
        return dict(status="NOT_MEASURED", reason="no usable position/reference samples"), trace
    gaps = [b["t"]-a["t"] for a,b in zip(trace[:-1],trace[1:])]
    if any(g <= 0 for g in gaps):
        audit.append(dict(vehicle=name, error="nonincreasing sample observation clock"))
    duration = identity.get("duration_s", setup.get("duration_s"))
    covered = bool(finite_number(duration) and trace[0]["t"] <= 0 and trace[-1]["t"] >= duration
                   and all(g > 0 for g in gaps))
    # Finite samples always have descriptive statistics. A full-window integral
    # is intentionally absent unless the observations actually bracket [0,T].
    integral = None
    if covered:
        total = 0.
        for a,b in zip(trace[:-1],trace[1:]):
            lo, hi = max(0.,a["t"]), min(duration,b["t"])
            if hi <= lo:
                continue
            e0, e1 = a["position_error_m"]**2, b["position_error_m"]**2
            alpha0, alpha1 = (lo-a["t"])/(b["t"]-a["t"]), (hi-a["t"])/(b["t"]-a["t"])
            total += (hi-lo)*(e0+(e1-e0)*alpha0+e0+(e1-e0)*alpha1)/2.
        integral = total/duration
    boundary_ratio = None
    if len(trace)>1 and gaps and all(g>0 for g in gaps):
        sa,sb=trace[0].get("sim_timestamp_ns"),trace[-1].get("sim_timestamp_ns")
        if finite_number(sa) and finite_number(sb) and sb>sa:
            boundary_ratio=(sb-sa)/1e9/(trace[-1]["t"]-trace[0]["t"])
    # Clock validation must use active request observations only. Paused
    # endpoint readback latency is explicitly not active-flight wall time.
    active=[r for r in trace if r.get("observation_role")=="active_slot"]
    ratio=None
    if len(active)>1:
        sa,sb=active[0].get("sim_timestamp_ns"),active[-1].get("sim_timestamp_ns")
        delta=active[-1]["t"]-active[0]["t"]
        if finite_number(sa) and finite_number(sb) and sb>sa and delta>0:
            ratio=(sb-sa)/1e9/delta
    errors = [r["position_error_m"] for r in trace]
    report = dict(status="MEASURED_SAMPLES", samples=len(trace), first_sample_s=trace[0]["t"],
                  last_sample_s=trace[-1]["t"], requested_duration_s=duration, window_covered=covered,
                  full_window_tracking_mse_m2=integral,
                  full_window_tracking_rmse_m=math.sqrt(integral) if integral is not None else None,
                  observed_position_error_m=stats(errors),
                  observed_sample_rmse_m=math.sqrt(sum(e*e for e in errors)/len(errors)),
                  observed_velocity_error_mps=stats([r["velocity_error_mps"] for r in trace if "velocity_error_mps" in r]),
                  max_observation_gap_ms=max(gaps)*1000 if gaps else None,
                  sim_wall_ratio=ratio, sim_clock_observed_valid=ratio is not None and .95<=ratio<=1.05,
                  sim_wall_ratio_scope="first_to_last_active_slot_state_RPC_start_proxies_only",
                  sim_wall_ratio_with_paused_proxies=boundary_ratio,
                  reference_scope=sorted(set(r["reference_scope"] for r in trace)),
                  max_tracking_error_reconstruction_difference_m=max(
                      [r["tracking_error_reconstruction_abs_m"] for r in trace if "tracking_error_reconstruction_abs_m" in r] or [0.]),
                  boundary_warnings=boundary_warnings,
                  full_window_tracking_scope="Piecewise-linear integral of sampled squared errors, with explicit paused boundary proxies when present; not continuous-physics ground truth")
    return report,trace


def gate_report(trace, layout, setup, max_gap_ms=100.):
    if not layout or len(trace)<2:
        return dict(status="NOT_MEASURED", reason="gate geometry or two sampled states missing", gates=[])
    width=float(setup.get("gate_center_width_m",4.))
    height=float(setup.get("gate_center_height_m",4.))
    thickness=float(setup.get("gate_thickness_m",.15))
    if min(width,height,thickness)<=0:
        raise ValueError("invalid physical gate dimensions")
    results=[]
    for gate in layout.get("gates",[]):
        events=[]
        for a,b in zip(trace[:-1],trace[1:]):
            crossing=segment_crossing(a,b,gate)
            if crossing is not None:
                crossing["point_clearance_margin_m"]=min(width/2-thickness/2-.25-abs(crossing["lateral_offset_m"]),
                                                        height/2-thickness/2-.25-abs(crossing["vertical_offset_m"]))
                crossing["body_margin_m"]=min(width/2-thickness/2-.525-abs(crossing["lateral_offset_m"]),
                                             height/2-thickness/2-.525-abs(crossing["vertical_offset_m"]))
                crossing["point_valid"]=crossing["point_clearance_margin_m"]>=0
                crossing["body_valid"]=crossing["body_margin_m"]>0
                crossing["telemetry_gap_acceptable"]=0<crossing["observation_gap_ms"]<=max_gap_ms
                events.append(crossing)
        first=events[0] if events else None
        results.append(dict(gate=gate["index"], crossed=bool(first), first_crossing=first,
                            forward_crossings_observed=len(events), all_forward_crossings=events))
    return dict(status="SAMPLED_SEGMENT_INTERSECTION", expected_gates=len(results),gates=results,
                all_point_crossings_observed=bool(results) and all(r["crossed"] and r["first_crossing"]["point_valid"] for r in results),
                all_body_crossings_observed=bool(results) and all(r["crossed"] and r["first_crossing"]["body_valid"] for r in results),
                all_crossing_gaps_acceptable=bool(results) and all(r["crossed"] and r["first_crossing"]["telemetry_gap_acceptable"] for r in results),
                scope="Exact forward segment-plane interpolation of logged samples; not continuous-physics swept-body certification",
                point_clearance_m=.25, body_radius_plus_clearance_m=.525, max_crossing_gap_ms=max_gap_ms)


def request_counts(rows):
    statuses=collections.Counter(str(r.get("status","missing")) for r in rows)
    errors=collections.Counter(str(r["error"]) for r in rows if r.get("error"))
    collisions=sum(bool(r.get("collision_object")) or "collision" in str(r.get("error","")) for r in rows)
    late=sum(r.get("status")=="command_return_late" or (finite_number(r.get("command_return_us")) and
             finite_number(r.get("actuation_deadline_us")) and r["command_return_us"]>=r["actuation_deadline_us"]) for r in rows)
    return dict(recorded_slots=len(rows), status=dict(sorted(statuses.items())), errors=dict(sorted(errors.items())),
                delivered_status=statuses["delivered"], delivered_without_recorded_error=sum(r.get("status")=="delivered" and not r.get("error") for r in rows),
                input_range_rejected=sum(r.get("status")=="input_range_rejected" or r.get("dpu_status")==3 for r in rows),
                collision_events=collisions, late_command_returns=late,
                dropped_slots=sum(str(r.get("status","")).startswith("dropped_") for r in rows),
                retired_unreleased=statuses["not_released_vehicle_retired"],
                command_delivery_unknown=sum(bool(r.get("command_delivery_unknown")) or
                    (r.get("command_may_have_been_sent") is True and r.get("command_sent") is not True) for r in rows),
                note="Collision/error may coexist with delivered status; these counts intentionally overlap.")


def write_csv(path, rows):
    keys=sorted(set(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(",",":"),allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def analyze_run(folder):
    audit=[]
    export_warnings=[]
    identity=load_json(folder/"execution_identity.json",{})
    setup=load_json(folder/"prepared_source.json",{})
    failure=load_json(folder/"failure.json")
    original_summary=load_json(folder/"summary.json")
    initial_states=load_json(folder/"initial_states.json",[])
    final_states=load_json(folder/"final_states.json",[])
    path=folder/"requests.jsonl"
    if not path.is_file():
        path=folder/"events.jsonl"
    rows=load_rows(path,audit) if path.is_file() else []
    partial_final_export=False
    final_export_rows=len(rows) if path.name=="requests.jsonl" else None
    journal_rows=None
    if not path.is_file():
        audit.append(dict(error="neither requests.jsonl nor events.jsonl exists"))
    if (folder/"requests.jsonl").is_file() and (folder/"events.jsonl").is_file():
        journal_audit=[]
        journal=load_rows(folder/"events.jsonl",journal_audit)
        journal_rows=len(journal)
        audit.extend(journal_audit)
        final_set={json.dumps(r,sort_keys=True) for r in rows}
        journal_set={json.dumps(r,sort_keys=True) for r in journal}
        if final_set!=journal_set:
            warning=dict(error="final requests and continuous journal differ",final_export_rows=len(rows),journal_rows=len(journal))
            # The continuous journal is authoritative for a failed final export
            # only when all exported rows are identical journal rows. A failed
            # or empty export must not hide already completed actual requests.
            if failure is not None and final_set<journal_set:
                export_warnings.append(warning)
                partial_final_export=True
                rows=journal
                path=folder/"events.jsonl"
            else:
                audit.append(warning)
    timing,timing_rows=phase_values(rows,audit)
    names=sorted(set(r.get("vehicle") for r in rows if isinstance(r.get("vehicle"),str))|set(setup.get("layouts",{})))
    vehicles={}
    traces=[]
    for name in names:
        subset=[r for r in rows if r.get("vehicle")==name]
        t,trace=tracking(subset,identity,setup,name,audit,initial_states,final_states)
        gates=gate_report(trace,setup.get("layouts",{}).get(name),setup)
        counts=request_counts(subset)
        covered=t.get("window_covered",False)
        duration,period=identity.get("duration_s"),identity.get("period_ms")
        expected=math.ceil(duration*1000/period) if finite_number(duration) and finite_number(period) and period>0 else None
        cycles=sorted(r["cycle"] for r in subset if isinstance(r.get("cycle"),int))
        release_complete=expected is not None and cycles==list(range(expected))
        vtiming,_=phase_values(subset,[])
        endpoints=[r for r in final_states if r.get("vehicle")==name]
        boundary_errors=[r["error"] for r in endpoints if r.get("error")]
        boundary_collisions=sum(r.get("collision_new") is True for r in endpoints)
        vehicles[name]=dict(counts=counts,tracking=t,gate_crossings=gates,timing_ms=vtiming,
            final_snapshot_errors=boundary_errors,final_snapshot_new_collisions=boundary_collisions,
            expected_planned_slots=expected,release_records_complete=release_complete,
            timing_passed=bool(subset) and release_complete and counts["delivered_without_recorded_error"]==len(subset) and not counts["late_command_returns"],
            flight_passed_observed=bool(covered and gates.get("all_body_crossings_observed") and gates.get("all_crossing_gaps_acceptable")
                and t.get("sim_clock_observed_valid") and not counts["collision_events"] and not counts["errors"] and not failure
                and not boundary_errors and not boundary_collisions))
        for row in trace:
            row["vehicle"]=name
            traces.append(row)
    files={}
    for p in [path,folder/"requests.jsonl",folder/"events.jsonl",folder/"execution_identity.json",folder/"prepared_source.json",folder/"summary.json",folder/"failure.json",folder/"initial_states.json",folder/"final_states.json"]:
        if p.is_file():files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(schema="ps-flight-observational-analysis/v1",data=str(folder.resolve()),
        input_log=str(path),used_partial_journal=path.name=="events.jsonl",failure=failure,
        partial_final_export=partial_final_export,export_warnings=export_warnings,
        final_export_rows=final_export_rows,journal_rows=journal_rows,
        runtime_summary=original_summary,execution_identity=identity,sha256=files,
        initial_states=initial_states,final_states=final_states,
        audit_errors=audit,audit_passed=not audit,counts=request_counts(rows),timing_ms=timing,
        pure_dpu_hardware_time_ms=None,vehicles=vehicles,
        complete_flight_observed=bool(vehicles) and not audit and all(v["flight_passed_observed"] for v in vehicles.values()),
        caveats=["DNNDK interval is DPU call wall time, NOT pure hardware execution time.",
                 "Command return is an RPC ping proxy, NOT physical actuation confirmation.",
                 "Reference timing is PS sample-start wall time, matching the C++ runtime; simulation time is separate.",
                 "Sample error statistics are descriptive even for early failures; do not rank failed short runs by lower sample RMSE.",
                 "Full-window tracking MSE is null without observed states bracketing [0,duration].",
                 "Initial/final paused snapshots, if supplied, are explicit boundary timing proxies; their observed_us is not an exact physics-state sampling time.",
                 "Gate intersections interpolate recorded states only; no continuous swept-body safety claim."])
    return report,timing_rows,traces


def markdown(report):
    lines=["# Native PS control observational report","",
           "This report separates transport/timing, sampled tracking, and flight evidence. A delivered command is not a flight-success claim.",""]
    if report.get("label"):
        lines.extend(["Campaign classification: `{}`.".format(report["label"]),""])
    for run in report["runs"]:
        lines.extend(["## "+run["data"],"", "Audit: `{}`; observed complete flight: `{}`.".format("PASS" if run["audit_passed"] else "FAIL",run["complete_flight_observed"]),
                      "", "Statuses: `{}`.".format(json.dumps(run["counts"]["status"],sort_keys=True)),""])
        if run["failure"]:lines.extend(["Preserved runtime failure: `{}`.".format(json.dumps(run["failure"],sort_keys=True)),""])
        if run["partial_final_export"]:lines.extend(["Final export is incomplete: analyzed {} actual continuous-journal rows instead of {} final-export rows. This is a preserved failed run, not formal completion.".format(run["journal_rows"],run["final_export_rows"]),""])
        lines.extend(["| PS phase (ms) | Samples | P50 | P95 | P99 | Maximum observed |","| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name,_,_ in PHASES:
            s=run["timing_ms"][name]
            if s["count"]:lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(name,s["count"],s["p50"],s["p95"],s["p99"],s["max"]))
        lines.append("")
        for name,v in sorted(run["vehicles"].items()):
            t=v["tracking"]
            lines.extend(["### "+name,"", "Recorded release window complete: `{}`. Tracking observation window covered: `{}`. Timing passed: `{}`.".format(v["release_records_complete"],t.get("window_covered",False),v["timing_passed"]),
                "", "Sample tracking RMSE (m): `{}`; full-window tracking RMSE (m): `{}`; sim/wall ratio: `{}`.".format(t.get("observed_sample_rmse_m"),t.get("full_window_tracking_rmse_m"),t.get("sim_wall_ratio")),
                "", "Gate report scope: `{}`.".format(v["gate_crossings"].get("status")),""])
    lines.extend(["## Interpretation limits",""])
    for caveat in report["runs"][0]["caveats"] if report["runs"] else []:lines.append("- "+caveat)
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,nargs="+",type=Path,help="One or more native runtime output directories")
    parser.add_argument("--output",required=True,type=Path,help="NEW analysis directory; never overwrite raw/previous results")
    parser.add_argument("--label",default=None,help="Optional campaign/diagnostic classification; does not change acceptance")
    args=parser.parse_args(argv)
    args.output.mkdir(parents=False,exist_ok=False)
    runs=[];timings=[];traces=[];gates=[]
    for folder in args.data:
        result,rows,track=analyze_run(folder)
        runs.append(result)
        for row in rows:row["run"]=str(folder);timings.append(row)
        for row in track:row["run"]=str(folder);traces.append(row)
        for name,vehicle in result["vehicles"].items():
            for gate in vehicle["gate_crossings"].get("gates",[]):
                item=dict(run=str(folder),vehicle=name,gate=gate["gate"],crossed=gate["crossed"],
                          forward_crossings_observed=gate["forward_crossings_observed"])
                item.update(gate["first_crossing"] or {})
                gates.append(item)
    report=dict(schema="ps-flight-analysis-collection/v1",runs=runs,label=args.label,
                audit_passed=all(r["audit_passed"] for r in runs),
                complete_flights_observed=sum(r["complete_flight_observed"] for r in runs))
    (args.output/"analysis.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    (args.output/"REPORT.md").write_text(markdown(report),encoding="utf-8")
    write_csv(args.output/"timing_per_request.csv",timings)
    write_csv(args.output/"tracking_samples.csv",traces)
    write_csv(args.output/"gate_crossings.csv",gates)
    print("PS_FLIGHT_ANALYSIS runs={} audit={} observed_complete_flights={} output={}".format(len(runs),"PASS" if report["audit_passed"] else "FAIL",report["complete_flights_observed"],args.output))
    return 0 if report["audit_passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
