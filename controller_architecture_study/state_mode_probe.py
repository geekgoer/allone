"""E1: paired recovery trials for six frozen FP32 supernet configurations."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import html
import json
from pathlib import Path
import time

import numpy as np

from .decision_dataset import ROOT, atomic_json, sha256


EXPORT = ROOT / "controller_architecture_study/results/incremental_supernet_237d_v4_grid5000/export"
OUTPUT = ROOT / "controller_architecture_study/results/state_mode_admissibility_v2"
PROTOCOL = {"position_tolerance_m": .3, "velocity_tolerance_mps": .6,
    "hold_seconds": .25, "minimum_progress_ratio": .8, "observation_period_s": .02,
    "gate_width_m": 4., "gate_height_m": 4., "gate_thickness_m": .15,
    "vehicle_radius_m": .20, "clearance_m": .25, "command_lease_s": 1.,
    "maximum_observation_gap_s": .10, "sim_wall_ratio_bounds": [.95, 1.05]}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_modes(metrics):
    """Six log-cost-spaced validation-frontier points, excluding >10x-best MSE."""
    rows = sorted((dict(depth=int(r["depth"]), width=int(r["width"]),
        effective_macs=int(r["effective_macs"]), validation_mse=float(r["validation"]["normalized_mse"]))
        for r in metrics["configurations"]), key=lambda r: (r["effective_macs"], r["validation_mse"]))
    best = min(r["validation_mse"] for r in rows)
    frontier, lowest = [], float("inf")
    for row in rows:
        if row["validation_mse"] <= 10*best and row["validation_mse"] < lowest:
            frontier.append(row)
            lowest = row["validation_mse"]
    if len(frontier) < 6:
        raise ValueError("fewer than six eligible validation-frontier modes")
    selected = [frontier[0], frontier[-1]]
    targets = np.linspace(np.log(frontier[0]["effective_macs"]), np.log(frontier[-1]["effective_macs"]), 6)[1:-1]
    for target in targets:
        selected.append(min((r for r in frontier if r not in selected),
            key=lambda r: abs(np.log(r["effective_macs"])-target)))
    selected.sort(key=lambda r: r["effective_macs"])
    for row in selected:
        row["name"] = f"d{row['depth']:02d}_w{row['width']:04d}"
    return selected, rows


def normal_frame(reference):
    velocity = reference.velocity[:2]
    speed = np.linalg.norm(velocity)
    if speed < .5:
        raise ValueError("recovery window must be on a moving reference, not terminal hover")
    tangent = velocity/speed
    derivative = (reference.acceleration[:2]-tangent*np.dot(tangent, reference.acceleration[:2]))/speed
    return np.r_[-tangent[1], tangent[0], 0.], np.r_[-derivative[1], derivative[0], 0.]


def risk_features(state, reference):
    """Signed clearance to a declared moving route tube, not an exact mesh distance."""
    normal, derivative = normal_frame(reference)
    error, dv = state.position-reference.position, state.velocity-reference.velocity
    lateral = float(np.dot(error, normal))
    lateral_rate = float(np.dot(dv, normal)+np.dot(error, derivative))
    radius = PROTOCOL["gate_width_m"]/2-PROTOCOL["vehicle_radius_m"]-PROTOCOL["clearance_m"]
    vertical_radius = PROTOCOL["gate_height_m"]/2-PROTOCOL["vehicle_radius_m"]-PROTOCOL["clearance_m"]
    margins = [radius-abs(lateral), vertical_radius-abs(error[2])]
    rates = [-np.sign(lateral)*lateral_rate, -np.sign(error[2])*dv[2]]
    active = int(np.argmin(margins))
    return {"h_m": float(margins[active]), "h_dot_mps": float(rates[active]),
        "active_boundary": ("lateral_positive" if lateral >= 0 else "lateral_negative") if active == 0
            else ("ned_down" if error[2] >= 0 else "ned_up"),
        "lateral_error_m": lateral}


def make_cases(duration):
    from .probe_airsim_three import load_layout
    from .fly_decision_dnn import StoredRoutePlanner
    from .fast_convex_preprocess import FastReferenceWindow

    cases, layouts = [], {}
    for geometry, route_index in (("straight", 0), ("turn", 3)):
        layout, digest = load_layout(48, route_index)
        planner = StoredRoutePlanner(layout)
        window = FastReferenceWindow(planner.trajectory, planner.config)
        times = np.linspace(1.5, planner.trajectory.duration-duration-.3, 150)
        refs = [window.window(float(t), horizon=1)[0] for t in times]
        if geometry == "straight":
            target = np.asarray(layout["gates"][0]["center"])-[2., 0., 0.]
            index = min(range(len(times)), key=lambda i: np.linalg.norm(refs[i].position-target))
        else:
            index = max(range(len(times)), key=lambda i: abs(float(np.cross(refs[i].velocity[:2], refs[i].acceleration[:2]))))
            if abs(float(np.cross(refs[index].velocity[:2], refs[index].acceleration[:2]))) < .5:
                raise ValueError("selected turn route has insufficient planar curvature")
        phase, ref = float(times[index]), refs[index]
        normal, _ = normal_frame(ref)
        radius = PROTOCOL["gate_width_m"]/2-PROTOCOL["vehicle_radius_m"]-PROTOCOL["clearance_m"]
        for margin in (.3, .6, 1.):
            for closing in (.4, 1.):
                position = ref.position+normal*(radius-margin)
                # Include rotating-frame motion so initial h_dot equals -closing.
                _, ndot = normal_frame(ref)
                velocity = ref.velocity+normal*(closing-np.dot(position-ref.position, ndot))
                cases.append({"id": f"{geometry}_h{margin:g}_v{closing:g}", "geometry": geometry,
                    "route_index": route_index, "environment_id": 48, "route_sha256": digest,
                    "reference_phase_s": phase, "position": position.tolist(), "velocity": velocity.tolist(),
                    "quaternion": ref.orientation.tolist(), "requested_h_m": margin, "requested_h_dot_mps": -closing,
                    "held_control": [float(np.linalg.norm(ref.acceleration-[0., 0., 9.80665])), 0., 0., 0.]})
        layouts[geometry] = layout
    return cases, layouts


def make_trials(cases, modes, repeats, seed):
    rng = np.random.default_rng(seed)
    order = [m["name"] for m in modes]
    rng.shuffle(order)
    trials = []
    for repeat in range(repeats):
        indices = rng.permutation(len(cases))
        rotated = order[repeat % len(order):]+order[:repeat % len(order)]
        for index in indices:
            for mode in rotated:
                trials.append({"id": f"r{repeat}_{cases[index]['id']}_{mode}",
                    "case": cases[index]["id"], "mode": mode, "repeat": repeat})
    return trials


def recovery_metrics(trace, commands, duration, *, error_kind=None):
    complete = bool(trace and trace[-1]["t"] >= duration)
    recovered_at, good_since = None, None
    for row in trace:
        if row["position_error_m"] <= PROTOCOL["position_tolerance_m"] and row["velocity_error_mps"] <= PROTOCOL["velocity_tolerance_mps"]:
            if good_since is None:
                good_since = row["t"]
            if row["t"]-good_since >= PROTOCOL["hold_seconds"]-1e-8 and row["t"] <= duration:
                recovered_at = row["t"] if recovered_at is None else recovered_at
        else:
            good_since = None
    ratio = None
    if len(trace) > 1 and trace[-1]["reference_arc_m"] > trace[0]["reference_arc_m"]:
        ratio = (trace[-1]["path_arc_m"]-trace[0]["path_arc_m"])/(trace[-1]["reference_arc_m"]-trace[0]["reference_arc_m"])
    sim_ratio = trace[-1]["t"]/trace[-1]["wall_s"] if trace and trace[-1]["wall_s"] > 0 else None
    gap = max(np.diff([r["wall_s"] for r in trace]), default=0.) if trace else None
    timing = bool(error_kind not in ("timing", "setup", "infrastructure", "interrupted")
        and sim_ratio is not None and PROTOCOL["sim_wall_ratio_bounds"][0] <= sim_ratio <= PROTOCOL["sim_wall_ratio_bounds"][1]
        and gap <= PROTOCOL["maximum_observation_gap_s"] and all(not r.get("timing_violation", False) for r in commands))
    safe_observed = bool(trace and all(r["h_m"] >= 0. and not r["collision"] for r in trace)
        and error_kind != "flight")
    progress = ratio is not None and ratio >= PROTOCOL["minimum_progress_ratio"]
    success = bool(complete and recovered_at is not None and safe_observed and progress)
    return {"window_covered": complete, "recovery_criterion_passed": recovered_at is not None,
        "recovery_confirmation_s": recovered_at, "observed_boundary_collision_passed": safe_observed,
        "progress_ratio": ratio, "control_success": success, "timing_comparable": timing,
        "joint_success": bool(success and timing and error_kind is None),
        "minimum_h_m": min((r["h_m"] for r in trace), default=None),
        "sim_wall_ratio": sim_ratio, "max_observation_gap_ms": None if gap is None else gap*1000,
        "timing_violations": sum(r.get("timing_violation", False) for r in commands),
        "delivery_deviations": sum(r.get("delivery_target_missed", False) for r in commands),
        "deadline_misses": sum(r.get("deadline_missed", False) for r in commands),
        "rpc_returns_after_deadline": sum(r.get("rpc_return_after_deadline", False) for r in commands)}


def summarize(plan, reports):
    rows = []
    for case in plan["cases"]:
        for mode in plan["modes"]:
            found = [r for r in reports if r["trial"]["case"] == case["id"] and r["trial"]["mode"] == mode["name"]]
            comparable = [r for r in found if r["timing_comparable"]]
            successes = sum(r["joint_success"] for r in found)
            # Three successes are a pilot observation, never a certified region.
            status = "unknown"
            if len(found) == plan["repeats"] and len(comparable) == len(found):
                status = "pilot_all_passed" if successes == len(found) else "observed_failure"
            rows.append({"case": case["id"], "mode": mode["name"], "effective_macs": mode["effective_macs"],
                "completed": len(found), "expected": plan["repeats"], "timing_comparable": len(comparable),
                "joint_successes": successes, "control_successes": sum(r["control_success"] for r in found),
                "timing_invalid": len(found)-len(comparable), "status": status})
    cheapest = []
    for case in plan["cases"]:
        eligible = [r for r in rows if r["case"] == case["id"] and r["status"] == "pilot_all_passed"]
        cheapest.append({"case": case["id"], "mode": min(eligible, key=lambda r: r["effective_macs"])["mode"] if eligible else None})
    static = [m for m in plan["modes"] if all(r["status"] == "pilot_all_passed" for r in rows if r["mode"] == m["name"])]
    baseline = min(static, key=lambda m: m["effective_macs"]) if static else None
    cost = {m["name"]: m["effective_macs"] for m in plan["modes"]}
    saving = (1-float(np.mean([cost[r["mode"]] for r in cheapest]))/baseline["effective_macs"]
        if baseline is not None and all(r["mode"] is not None for r in cheapest) else None)
    return {"scope": "E1 exploratory matched-state pilot; not an admissibility certificate or a runtime map",
        "completed_trials": len(reports), "expected_trials": len(plan["trials"]),
        "joint_successes": sum(r["joint_success"] for r in reports), "rows": rows,
        "failure_kinds": dict(Counter(r["error_kind"] or "criteria_not_met" for r in reports if not r["joint_success"])),
        "minimum_observed_passing_modes": cheapest,
        "cheapest_observed_all_states_static_mode": None if baseline is None else baseline["name"],
        "equal_state_weighted_MAC_saving_oracle_fraction": saving,
        "oracle_warning": "hindsight state-by-state choice ignores runtime costs; not deployable and not a significance test",
        "test_read": False}


def save_summary(output, plan, reports):
    summary = summarize(plan, reports)
    atomic_json(output/"summary.json", summary)
    with (output/"summary.csv.tmp").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    (output/"summary.csv.tmp").replace(output/"summary.csv")
    # Standalone SVG, no plotting dependency; counts and missingness remain explicit.
    width, height = 1080, 130+len(plan["cases"])*37
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        '<title>E1 recovery pilot: joint successes / completed trials; I = timing-invalid trials</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial,sans-serif" font-size="13" fill="#111111">',
        '<text x="20" y="24">E1 pilot: success / completed; I = timing-invalid; pending is not zero</text>']
    for i, mode in enumerate(plan["modes"]):
        svg.append(f'<text x="{310+i*122}" y="53" text-anchor="middle">{mode["name"]}</text>')
    for j, case in enumerate(plan["cases"]):
        y = 65+j*37
        svg.append(f'<text x="12" y="{y+21}">{html.escape(case["id"])}</text>')
        for i, mode in enumerate(plan["modes"]):
            row = next(r for r in summary["rows"] if r["case"] == case["id"] and r["mode"] == mode["name"])
            rate = row["joint_successes"]/row["completed"] if row["completed"] else None
            fill = "#dddddd" if rate is None else ("#bdd7e7" if rate == 1 else "#ffffff" if rate == 0 else "#deebf7")
            label = "pending" if rate is None else f'{row["joint_successes"]}/{row["completed"]} I={row["timing_invalid"]}'
            if row["completed"] and row["completed"] < row["expected"]:
                label += " partial"
            x = 252+i*122
            svg.extend([f'<rect x="{x}" y="{y}" width="116" height="32" fill="{fill}" stroke="#777777"/>',
                f'<text x="{x+58}" y="{y+21}" text-anchor="middle">{label}</text>'])
    svg.extend([f'<text x="20" y="{height-17}">Source: summary.csv; three paired repeats are exploratory, not a safety guarantee.</text>', '</g></svg>'])
    (output/"recovery_heatmap.svg").write_text("\n".join(svg), encoding="utf-8")
    return summary


class SupernetPool:
    def __init__(self, export, modes, device):
        import torch

        torch.set_num_threads(1)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
        self.torch, self.device = torch, device
        self.model = torch.jit.load(str(export/"controller_scripted.pt"), map_location=device).eval()
        self.modes = {m["name"]: m for m in modes}
        self.runtime = {"torch": torch.__version__, "numpy": np.__version__, "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "torchscript_optimized_execution": False}

    def predict(self, name, feature):
        mode, torch = self.modes[name], self.torch
        started = time.perf_counter()
        # Avoid new profiling/specialization work on a mode switch during flight.
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            x = torch.as_tensor(np.asarray(feature, dtype=np.float32)[None], device=self.device)
            action = self.model(x, mode["depth"], mode["width"]).cpu().numpy()[0].astype(float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise RuntimeError("invalid supernet physical output")
        return action, {"inference_ms": (time.perf_counter()-started)*1000}


def warmup_features(cases, layouts):
    from airsim_interface import RawDroneState
    from .fly_decision_dnn import StoredRoutePlanner
    from .fast_convex_preprocess import FastReferenceWindow, encode

    windows = {}
    for geometry, layout in layouts.items():
        planner = StoredRoutePlanner(layout)
        windows[geometry] = FastReferenceWindow(planner.trajectory, planner.config)
    features = {}
    for case in cases:
        state = RawDroneState(case["position"], case["quaternion"], case["velocity"], np.zeros(3), np.zeros(3))
        features[case["id"]] = encode(state, windows[case["geometry"]].window(case["reference_phase_s"]))
    return features


def build_plan(args):
    from config import load_calibration

    export = args.export_dir.resolve()
    report, spec = read_json(export/"export_report.json"), read_json(export/"model_spec.json")
    if (report["status"] != "PASS" or spec.get("input_contract") != "decision_convex_local_relative_pv_237d_v1"
            or spec.get("network_contract") != "causal_group_residual_depth_width_v2"):
        raise ValueError("expected the frozen FP32 causal 237D supernet export")
    artifacts = {}
    for name in ("controller_scripted.pt", "normalization.json", "model_spec.json", "offline_metrics.json"):
        artifacts[name] = sha256(export/name)
        if artifacts[name] != report["artifact_sha256"][name]:
            raise ValueError(f"frozen artifact changed: {name}")
    modes, all_modes = select_modes(read_json(export/"offline_metrics.json"))
    cases, layouts = make_cases(args.duration)
    dependencies = [Path(__file__), Path(__file__).with_name("state_mode_flight.py"),
        Path(__file__).with_name("fast_convex_preprocess.py"), Path(__file__).with_name("fly_decision_dnn.py"),
        Path(__file__).with_name("probe_airsim_three.py"), ROOT/"control_adapter.py", ROOT/"airsim_interface.py",
        ROOT/"gate_visualization.py", ROOT/"config.py", ROOT/"state_adapter.py",
        ROOT/"falcon_paper_fidelity/motion_primitive.py"]
    plan = {"version": 2, "experiment": "state_mode_E1", "export_dir": str(export),
        "artifacts": artifacts, "checkpoint_sha256": report["checkpoint_sha256"],
        "modes": modes, "all_offline_modes": all_modes,
        "selection_rule": "six log-MAC-spaced validation-frontier points with MSE <=10x best; chosen before flight",
        "cases": cases, "repeats": args.repeats, "trials": make_trials(cases, modes, args.repeats, args.seed),
        "duration_s": args.duration, "period_ms": args.period_ms, "delivery_ms": args.delivery_ms,
        "delivery_tolerance_ms": args.delivery_tolerance_ms, "device": args.device,
        "host": args.host, "port": args.port, "vehicle": args.vehicle, "protocol": PROTOCOL,
        "calibration": asdict(load_calibration()), "test_read": False,
        "dependency_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in dependencies},
        "boundary": "virtual moving route tube; real gate collisions checked separately",
        "clock": "reference follows AirSim timestamps; release/dispatch follows host monotonic clock; ratio checked",
        "held_control": "reference feed-forward thrust, zero body rates; common to all modes in each case",
        "timestamps": "comparison target uses host command-send; RPC return logged separately; actuator time is unknown",
        "timing_policy": "target deviations mark trials incomparable but continue; expired outputs are not sent",
        "placement": "teleport physics and UE actor while paused; read both back; normal flight collisions remain checked",
        "pairing_limit": "observable kinematics matched; hidden simulator/controller state not bit-exact"}
    return plan, layouts


def completed_report(directory):
    paths = sorted(directory.glob("attempt_*/report.json"))
    for path in paths:
        report = read_json(path)
        if report["status"] == "finished":
            return report
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=EXPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--duration", type=float, default=3.)
    parser.add_argument("--period-ms", type=float, default=50.)
    parser.add_argument("--delivery-ms", type=float, default=20.)
    parser.add_argument("--delivery-tolerance-ms", type=float, default=8.)
    parser.add_argument("--max-trials", type=int, help="bound new trials this invocation; same manifest resumes later")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if (args.repeats < 1 or args.duration < 1. or not np.isfinite([args.duration, args.period_ms, args.delivery_ms,
            args.delivery_tolerance_ms]).all() or args.delivery_ms <= 0 or args.delivery_tolerance_ms < 0
            or args.delivery_ms+args.delivery_tolerance_ms >= args.period_ms or (args.max_trials is not None and args.max_trials < 1)):
        parser.error("use positive finite timing, duration>=1s, and delivery+tolerance<period")
    args.output = args.output.resolve()
    if args.summarize_only:
        plan = read_json(args.output/"setup.json")["plan"]
        reports = [r for t in plan["trials"] if (r := completed_report(args.output/t["id"])) is not None]
        print(json.dumps(save_summary(args.output, plan, reports), indent=2))
        return
    plan, layouts = build_plan(args)
    if args.plan_only:
        print(json.dumps({"modes": plan["modes"], "cases": plan["cases"], "trials": len(plan["trials"]),
            "minimum_flight_minutes": len(plan["trials"])*args.duration/60, "output": str(args.output)}, indent=2))
        return
    from .state_mode_flight import flight
    from .fly_convex_policy import load_airsim_client
    from airsim_interface import require_airsim

    pool = SupernetPool(args.export_dir, plan["modes"], args.device)
    setup = {"plan": plan, "runtime": pool.runtime}
    args.output.mkdir(parents=True, exist_ok=True)
    setup_path = args.output/"setup.json"
    if setup_path.exists():
        if not args.resume or read_json(setup_path) != setup:
            raise ValueError("existing setup differs or --resume missing; use a new output directory")
    elif any(args.output.iterdir()):
        raise ValueError("nonempty output without setup.json; use a new output directory")
    else:
        atomic_json(setup_path, setup)
    reports = [r for t in plan["trials"] if (r := completed_report(args.output/t["id"])) is not None]
    save_summary(args.output, plan, reports)
    pending = [t for t in plan["trials"] if t["id"] not in {r["trial"]["id"] for r in reports}]
    if not pending:
        print(f"All {len(reports)} trials already saved. Summary: {args.output/'summary.json'}")
        return
    features = warmup_features(plan["cases"], layouts)
    timings = {m["name"]: [] for m in plan["modes"]}
    # Exercise real inputs in the same paired mode order as flight, twice.
    for repeat in range(2):
        for trial in plan["trials"][:len(plan["cases"])*len(plan["modes"])]:
            _, measured = pool.predict(trial["mode"], features[trial["case"]])
            if repeat:
                timings[trial["mode"]].append(measured["inference_ms"])
    atomic_json(args.output/"warmup.json", {"real_case_inputs": True, "switch_order_matched": True,
        "post_warmup_ms": timings, "runtime": pool.runtime})
    for name, values in timings.items():
        print(f"Warm {name}: max inference={max(values):.2f}ms (real case inputs)", flush=True)
        if max(values) >= args.period_ms:
            raise RuntimeError("warmed inference exceeds the job deadline; choose a viable device/timing in a new output")
    load_airsim_client()
    client = require_airsim().MultirotorClient(ip=args.host, port=args.port, timeout_value=3.)
    client.confirmConnection()
    if args.vehicle not in client.listVehicles() or client.simIsPause():
        raise RuntimeError("manual Play needs the requested vehicle and an unpaused simulation")
    foreign = [n for n in client.simListSceneObjects("Falcon.*Gate.*") if not n.startswith("FalconModeGate_")]
    if foreign:
        raise RuntimeError("Stop/Play to clear previous generated gates before E1; unrelated gates were left untouched")
    cases = {c["id"]: c for c in plan["cases"]}
    for trial in pending[:args.max_trials]:
        directory = args.output/trial["id"]
        attempt = directory/f"attempt_{len(list(directory.glob('attempt_*')))+1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        print(f"[{len(reports)+1}/{len(plan['trials'])}] {trial['id']}", flush=True)
        case = cases[trial["case"]]
        report = flight(client, pool, layouts[case["geometry"]], case, trial, args, attempt)
        if report["status"] == "finished":
            reports.append(report)
        save_summary(args.output, plan, reports)
        print(f"  recovery={report['control_success']} timing={report['timing_comparable']} joint={report['joint_success']} "
            f"target_deviations={report['delivery_deviations']} deadline_misses={report['deadline_misses']} error={report['error']}", flush=True)
        if report["status"] != "finished":
            raise SystemExit(130 if report["error_kind"] == "interrupted" else 1)
    summary = save_summary(args.output, plan, reports)
    print(f"Saved {summary['completed_trials']}/{summary['expected_trials']}; joint successes={summary['joint_successes']}")
    print(f"Summary: {args.output/'summary.json'}\nHeatmap: {args.output/'recovery_heatmap.svg'}")


if __name__ == "__main__":
    main()
