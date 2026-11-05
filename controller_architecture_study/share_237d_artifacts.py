"""Package and check the original artifacts for the single residual237D handoff."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("controller_architecture_study/results")
RUN = "decision_convex_residual4_v1/runs/residual_mlp_h21_w1024_seed0"
FILES = (
    "decision_convex_96k_v1/manifest.json",
    "decision_convex_96k_v1/dnn_dataset/split_manifest.json",
    "decision_convex_96k_v1/dnn_dataset/normalization.json",
    "decision_convex_recovery_v1/manifest.json",
    "decision_convex_recovery_v1/data_audit.json",
    "decision_convex_recovery_v1/datasets/features_recovery/split_manifest.json",
    "decision_convex_recovery_v1/datasets/features_recovery/normalization.json",
    "decision_convex_recovery_v1/datasets/features_recovery/validation.npz",
    RUN + "/config.json", RUN + "/completion.json", RUN + "/best_model.pt",
    "decision_convex_96k_v1/routes/env48_route00.json",
)
MANIFEST = "237d_handoff_manifest.json"
GUIDE = Path("controller_architecture_study/docs/SHARE_237D_ARTIFACTS_20261105.md")


def stream_digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return stream_digest(stream)


def check(manifest=None):
    if manifest is not None:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        expected_files = {(RESULTS / name).as_posix() for name in FILES}
        if set(saved["files"]) != expected_files:
            raise ValueError("unexpected handoff file set")
        for name, info in saved["files"].items():
            path = ROOT / name
            if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                raise ValueError(f"handoff artifact changed: {name}")

    import numpy as np
    import torch
    from .dynamic_probe_common import BASE, DATA, provenance
    from .fly_convex_policy import ConvexDNN

    torch.set_num_threads(1)
    evidence = provenance(["residual21x1024"], validation=True)
    route = "env48_route00"
    collection = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    if not collection["complete"] or digest(BASE / "routes" / (route + ".json")) != collection["chunks"][route]["route_sha256"]:
        raise ValueError("flight route changed")
    controller = ConvexDNN(ROOT / RESULTS / RUN / "best_model.pt", BASE, "cpu")
    with np.load(DATA / "validation.npz", allow_pickle=False) as archive:
        features, row_ids = archive["X"][:32], archive["row_id"][:32]
        shape = archive["X"].shape
    if shape != (9600, 237) or features.shape != (32, 237) or row_ids.shape != (32,):
        raise ValueError("unexpected frozen validation shapes")
    x = (features.astype(np.float32) - controller.norm["x_mean"]) / controller.norm["x_std"]
    with torch.inference_mode():
        physical = controller.model(torch.from_numpy(x)).numpy() * controller.norm["u_std"] + controller.norm["u_mean"]
    if physical.shape != (32, 4) or not np.isfinite(physical).all():
        raise ValueError("invalid residual237D inference outputs")
    return dict(status="PASS", input_contract=evidence["input_contract"], input_dim=237,
        validation_shape=list(shape), inference_rows=32, output_shape=list(physical.shape),
        checkpoint_sha256=evidence["models"]["residual21x1024"]["checkpoint_sha256"],
        normalization_sha256=evidence["normalization_sha256"],
        validation_sha256=evidence["validation_sha256"], test_read=False, flight_run=False)


def package(output):
    if output.suffix.lower() != ".zip":
        raise ValueError("output must be a .zip file")
    checksum = output.with_suffix(".zip.sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError("handoff version already exists")
    evidence = check()
    files = {}
    for name in FILES:
        relative = RESULTS / name
        path = ROOT / relative
        files[relative.as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
    manifest = dict(version="residual237d_collaboration_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="one independent residual21x1024 controller, validation and default flight route",
        raw_files_preserved=True, files=files, preflight=evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
        archive.write(ROOT / GUIDE, "READ_237D_HANDOFF.md")
        for name in files:
            archive.write(ROOT / name, name)
    # Check the compressed payload independently before handing it to a colleague.
    with zipfile.ZipFile(output) as archive:
        for name, info in files.items():
            with archive.open(name) as stream:
                if stream_digest(stream) != info["sha256"]:
                    raise ValueError(f"archived bytes changed: {name}")
    checksum.write_text(digest(output) + "  " + output.name + "\n", encoding="utf-8")
    return dict(archive=str(output.resolve()), bytes=output.stat().st_size, artifact_files=len(files),
        sha256_file=str(checksum.resolve()), preflight=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package")
    pack.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("check")
    verify.add_argument("--manifest", type=Path, default=ROOT / MANIFEST)
    args = parser.parse_args()
    result = package(args.output) if args.command == "package" else check(args.manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
