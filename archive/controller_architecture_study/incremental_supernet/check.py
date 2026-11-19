"""Read-only source/data checks plus a few real 237D inference queries; no training."""

import json

import torch

from .common import (DEFAULT_CONFIG, DEFAULT_RUN, atomic_json, grid, load_teacher,
                     model_from_config, sha256, verify_assets)
from .model import IncrementalContext, PhysicalController


def main():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    norm, data, _ = verify_assets(config)
    teacher = load_teacher(config)
    model = model_from_config(config).eval()
    model.initialize_from_teacher(teacher)
    x = data["validation"][0][:2]
    with torch.inference_mode():
        output = model.outputs(model.features(x), grid(model))
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise RuntimeError("non-finite masked initialization")
        max_error = 0.
        for d,w in ((1,128),(9,384),(17,768),(33,1024)):
            max_error = max(max_error,float((output[(d,w)]-model(x,d,w)).abs().max()))
        cache_error = 0.
        for sequence in (((9,128),(33,128),(33,1024)),((1,1024),(17,1024),(33,1024))):
            context = IncrementalContext(model,x,"check",config["teacher_sha256"])
            for d,w in sequence:
                cached,nodes = context.grow(d,w,sample_id="check",model_version=config["teacher_sha256"])
            cache_error = max(cache_error,float((cached-model(x)).abs().max()))
        scripted = torch.jit.script(PhysicalController(model,norm))
        raw = x*torch.from_numpy(norm["x_std"]) + torch.from_numpy(norm["x_mean"])
        script_error = float((scripted(raw,33,1024) - PhysicalController(model,norm)(raw,33,1024)).abs().max())
    if max_error > 5e-5 or cache_error > 5e-5 or script_error > 5e-5:
        raise RuntimeError(f"237D inference mismatch: grid={max_error}, cache={cache_error}, script={script_error}")
    report = {"status":"PASS", "input_dim":237,"configurations":len(output),"depth_count":17,"width_count":8,
              "full_grid_vs_selected_max_abs":max_error,"incremental_orders_max_abs":cache_error,
              "torchscript_max_abs":script_error,"teacher_sha256":config["teacher_sha256"],
              "training_started":False,"test_read":False,
              "source_file_sha256":{name:sha256(DEFAULT_CONFIG.parent.parent/"incremental_supernet"/name)
                                    for name in ("model.py","train.py","export.py")},
              "note":"Real-shape initialization/API check, not trained accuracy or flight acceptance."}
    path = DEFAULT_RUN.parent / "incremental_supernet_checks/real_237d_check.json"
    atomic_json(path,report)
    print(json.dumps(report,indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
