"""Freeze heterogeneous AirSim cases and a separate 12-vehicle PS control build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from controller_architecture_study.ip37_safety_cases import audit, make_case


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_one(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Expected exactly one source anchor: " + before[:70])
    return source.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new directory")
    manifest = json.loads((args.base_control / "manifest.json").read_text(encoding="utf-8"))
    if not all(digest(args.base_control / name) == sha for name, sha in manifest.items()):
        raise ValueError("Frozen control004 source hash mismatch")
    args.output.mkdir(parents=True)
    cases = args.output / "cases"
    cases.mkdir()
    case_summary = []
    for width in (3.0, 3.4, 3.8):
        case = make_case(0, width, vehicles=12)
        checked = audit(case)
        if not checked["passed"] or case["duration_s"] != 26.:
            raise ValueError("Reference geometry or finite duration failed")
        filename = f"case_w{width:.1f}.json"
        (cases / filename).write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        (cases / f"audit_w{width:.1f}.json").write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
        case_summary.append({"width_m": width, "filename": filename, "sha256": digest(cases / filename),
                             "roles": {name: role["role"] for name, role in case["roles"].items()},
                             "gate_counts": {name: len(layout["gates"]) for name, layout in case["layouts"].items()},
                             "max_reference_speed_mps": max(v["max_sampled_speed_mps"] for v in checked["vehicles"].values()),
                             "max_reference_acceleration_mps2": max(v["max_sampled_acceleration_mps2"] for v in checked["vehicles"].values())})

    source = args.output / "control_source"
    shutil.copytree(args.base_control, source)
    runtime = source / "adaptive_runtime.cpp"
    code = runtime.read_text(encoding="utf-8")
    code = replace_one(code,
        "        Json initial_states=Json::array(),names=Json::array();",
        """        Json initial_states=Json::array(),names=Json::array();
        const auto& case_source=setup.at("original_case");
        const double clear_width=case_source.value("net_width_m",
            case_source.at("gate_center_width_m").get<double>()-.15);
        const double clear_height=case_source.at("gate_center_height_m").get<double>()-.15;
        if(clear_width<=0.||clear_height<=0.||
           std::abs(clear_width+.15-case_source.at("gate_center_width_m").get<double>())>1e-6)
            throw std::runtime_error("Case aperture and physical gate geometry disagree");""")
    code = replace_one(code,
        "v->analyzer.reset(new f::GateHoldAnalyzer(v->gates,3.85,f::HoldBounds(),3.85));",
        "v->analyzer.reset(new f::GateHoldAnalyzer(v->gates,clear_width,f::HoldBounds(),clear_height));")
    code = replace_one(code,
        "if(vehicles.empty()||vehicles.size()>2)throw std::runtime_error(\"Stage2 accepts one or two prepared vehicles only\");",
        "if(vehicles.empty()||vehicles.size()>12)throw std::runtime_error(\"Capacity study accepts 1..12 prepared vehicles\");")
    runtime.write_text(code, encoding="utf-8")
    base_config = json.loads((args.base_control / "config_dual_001_repair.json").read_text(encoding="utf-8"))
    for period in (20, 30, 40, 50, 80):
        config = dict(base_config)
        config["period_ms"] = float(period)
        config["periods_ms"] = [20., 30., 40., 50., 60., 80.]
        config["enable_skip"] = False
        config["capacity_study"] = "frozen heterogeneous route seed0; no synthetic inference delay"
        (source / f"config_edge_T{period}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    new_manifest = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*")
                    if p.is_file() and p.name != "manifest.json"}
    (source / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    with tarfile.open(str(source) + ".tar.gz", "w:gz") as archive:
        for p in source.rglob("*"):
            if p.is_file():
                archive.add(p, arcname=p.relative_to(source).as_posix())
    result = {"base_manifest_sha256": digest(args.base_control / "manifest.json"),
              "source_manifest_sha256": digest(source / "manifest.json"),
              "source_files": len(new_manifest), "cases": case_summary,
              "scope": "offline reference geometry only; flight and scheduling capacity unverified"}
    (args.output / "screen_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
