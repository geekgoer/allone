"""Read all retained v6 trials, including failures; never starts a flight."""

import argparse
import json
from pathlib import Path

import numpy as np

from .decision_dataset import atomic_json, sha256


def summarize(root):
    rows = []
    for path in sorted(root.glob("n*_net2/smoke_*/setup.json")):
        setup = json.loads(path.read_text())
        current = all(sha256(Path(__file__).with_name(n)) == digest for n,digest in setup["source_sha256"].items())
        for policy in ("static","adaptive"):
            directory = path.parent/policy
            row = {"run":str(directory.resolve()),"vehicles":setup["vehicles"],"policy":policy,
                "current_source_matches":current,"cost_profile_sha256":setup.get("cost_profile_sha256"),
                "report_present":(directory/"report.json").exists()}
            if row["report_present"]:
                report = json.loads((directory/"report.json").read_text())
                requests = json.loads((directory/"requests.json").read_text())
                fields = {"capture_ms":("sample_io_finished_s","sample_submitted_s"),
                    "feature_ms":("feature_done_s","frontend_start_s"),
                    "command_rpc_ms":("rpc_return_s","io_started_s"),
                    "command_with_dispatch_ms":("rpc_return_s","io_submitted_s"),
                    "sample_to_command_ms":("rpc_return_s","sample_s")}
                row.update({k:report[k] for k in ("accepted","total_J","collisions","abort","error",
                    "applied","actual_timing_violation_observed","segment_resumptions","model_switches",
                    "input_cost_exceedances","sample_io_cost_exceedances","io_cost_exceedances","command_cost_exceedances")})
                row["timing"] = {k:{"count":len(values),"p95":float(np.percentile(values,95)),"max":max(values)}
                    for k,(end,start) in fields.items()
                    if (values:=[1000*(r[end]-r[start]) for r in requests if end in r and start in r])}
            rows.append(row)
    result = {"trials":rows,"all_failures_retained":True,"FPGA_verified":False,
        "scope":"finite nominal CPU integration; not an optimized FALCON baseline or safety-advantage experiment"}
    atomic_json(root/"stage_summary.json",result)
    for r in rows:
        print(f"N={r['vehicles']} {r['policy']:8} current={r['current_source_matches']} "
            f"accepted={r.get('accepted')} J={r.get('total_J')} abort={r.get('abort')} {r['run']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    summarize(parser.parse_args().root)
