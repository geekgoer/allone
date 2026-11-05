"""Resolve moved artifacts while preserving the frozen collection hashes."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from .decision_dataset import ROOT, sha256


def project_path(value):
    """Rebase recorded project paths, including Windows paths on Linux."""
    windows = PureWindowsPath(str(value))
    path = windows if windows.drive or "\\" in str(value) else PurePosixPath(str(value))
    if not path.is_absolute():
        result = ROOT.joinpath(*path.parts).resolve()
    else:
        native = Path(value)
        if native.is_absolute() and (native.resolve().is_relative_to(ROOT.resolve())
                or "controller_architecture_study" not in path.parts):
            return native.resolve()
        parts = path.parts
        if "controller_architecture_study" not in parts:
            raise ValueError(f"cannot relocate a path outside the research project: {value}")
        result = ROOT.joinpath(*parts[parts.index("controller_architecture_study"):]).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError(f"artifact path escapes the project: {value}")
    return result


def verify_recovery_artifacts(data_dir, base):
    """Inference-only counterpart to the frozen, path-bound verify_arm.

    Dataset identity comes from collection hashes and split metadata. The
    original absolute base path remains intact as historical provenance.
    Keep this outside convex_recovery.py, whose own bytes are hash-bound.
    """
    from .convex_features import CONTRACT, KEPT
    from .convex_recovery import VERSION
    from .fly_convex_policy import verify_dataset

    source = verify_dataset(base, full=False)
    split = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
    output = data_dir.parent.parent
    if (split["version"] != VERSION or split["contract"] != CONTRACT
            or split["source_manifest_sha256"] != source["source_manifest_sha256"]
            or split["environment_ids"] != source["environment_ids"]
            or split["kept_original_indices"] != list(KEPT)
            or split["augmentation_manifest_sha256"] != sha256(output / "manifest.json")
            or json.loads((output / "data_audit.json").read_text(encoding="utf-8"))["status"] != "PASS"):
        raise ValueError("recovery source/contract/audit mismatch")
    config = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["config"]
    for name, expected in config["source_sha256"].items():
        if sha256(project_path(name)) != expected:
            raise ValueError(f"recovery source code changed: {name}")
    if sha256(data_dir / "normalization.json") != split["artifact_sha256"]["normalization.json"]:
        raise ValueError("recovery normalization changed")
    return split
