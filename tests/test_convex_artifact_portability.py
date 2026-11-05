import json
from pathlib import Path

import pytest

from controller_architecture_study import convex_artifacts as artifacts
from controller_architecture_study.convex_dataset import CONTRACT as BASE_CONTRACT, VERSION as BASE_VERSION
from controller_architecture_study.convex_features import CONTRACT, KEPT
from controller_architecture_study.convex_recovery import VERSION
from controller_architecture_study.decision_dataset import sha256
from controller_architecture_study.fly_convex_policy import verify_dataset


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def moved_data(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    base = tmp_path / "controller_architecture_study/results/base"
    recovery = tmp_path / "controller_architecture_study/results/recovery"
    data = recovery / "datasets/features_recovery"
    teacher = tmp_path / "controller_architecture_study/frozen_teacher.py"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"# exact frozen source bytes\n")
    code = {"controller_architecture_study\\frozen_teacher.py": sha256(teacher)}
    save(base / "manifest.json", {"config": {"source_sha256": code}})
    save(base / "dnn_dataset/normalization.json", {"base": True})
    norm_hash = sha256(base / "dnn_dataset/normalization.json")
    split = dict(version=BASE_VERSION, contract=BASE_CONTRACT, source_manifest_sha256=sha256(base / "manifest.json"),
        environment_ids={"train": [0], "validation": [1], "test": [2]}, artifact_sha256={"normalization.json": norm_hash})
    save(base / "dnn_dataset/split_manifest.json", split)
    save(recovery / "manifest.json", {"config": {"source_sha256": code}})
    save(recovery / "data_audit.json", {"status": "PASS"})
    save(data / "normalization.json", {"recovery": True})
    save(data / "split_manifest.json", {**split, "version": VERSION, "contract": CONTRACT,
        "base": "Z:\\original_machine\\controller_architecture_study\\results\\base",
        "kept_original_indices": list(KEPT), "augmentation_manifest_sha256": sha256(recovery / "manifest.json"),
        "artifact_sha256": {"normalization.json": sha256(data / "normalization.json")}})
    return base, data, teacher


def test_relocated_inference_checks_content_without_rewriting_originals(moved_data):
    base, data, _ = moved_data
    before = {p: p.read_bytes() for p in base.parent.rglob("*.json")}
    assert verify_dataset(base, full=False)["contract"] == BASE_CONTRACT
    assert artifacts.verify_recovery_artifacts(data, base)["contract"] == CONTRACT
    assert before == {p: p.read_bytes() for p in before}
    assert not (data / "train.npz").exists()
    assert not (data / "test.npz").exists()


@pytest.mark.parametrize("target", ["base_manifest", "augmentation_manifest", "normalization", "source"])
def test_relocation_still_rejects_changed_artifacts(moved_data, target):
    base, data, teacher = moved_data
    paths = {"base_manifest": base / "manifest.json", "augmentation_manifest": data.parent.parent / "manifest.json",
        "normalization": data / "normalization.json", "source": teacher}
    with paths[target].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        artifacts.verify_recovery_artifacts(data, base)


@pytest.mark.parametrize("recorded", [
    "F:\\FalconReproduction\\airsim_control\\controller_architecture_study\\results\\data",
    "F:/FalconReproduction/airsim_control/controller_architecture_study/results/data",
    "/home/colleague/old_repo/controller_architecture_study/results/data",
    "controller_architecture_study\\results\\data",
])
def test_recorded_paths_are_relative_to_current_checkout(tmp_path, monkeypatch, recorded):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    assert artifacts.project_path(recorded) == tmp_path / "controller_architecture_study/results/data"


def test_recorded_path_cannot_escape_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        artifacts.project_path("../elsewhere")


def test_native_absolute_custom_dataset_path_remains_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    external = Path(tmp_path.anchor) / "custom_dataset"
    assert artifacts.project_path(str(external)) == external.resolve()
