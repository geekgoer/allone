"""Build a self-contained Linux-server package for fixed-3000 DNN training.

The package intentionally contains only the prepared split data and the code
required to train it.  It excludes AirSim, ACADO, raw core/extension chunks
and previous experimental outputs.  SHA-256 hashes are included so the server
can verify that the uploaded files are unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "falcon_hybrid_160k_10ms_v4_server_train_3000"
DATASET_NAME = "dnn_dataset_hybrid_160k_10ms_v4_environment_balanced"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE / "server_transfer")
    parser.add_argument("--force", action="store_true", help="replace a previous package directory/archive")
    args = parser.parse_args()

    dataset = HERE / DATASET_NAME
    required_data = ("train.npz", "validation.npz", "test.npz", "normalization.json", "split_manifest.json")
    missing = [name for name in required_data if not (dataset / name).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared v3 dataset is incomplete: {missing}")

    output_root = args.output_root.resolve()
    package = output_root / PACKAGE_NAME
    archive = output_root / f"{PACKAGE_NAME}.zip"
    if package.exists() or archive.exists():
        if not args.force:
            raise FileExistsError(f"package exists: {package}; use --force to rebuild")
        if package.exists():
            shutil.rmtree(package)
        if archive.exists():
            archive.unlink()

    package.mkdir(parents=True)
    repository_root = HERE.parents[1]
    for name in ("train_dnn_controller.py", "dnn_model_utils.py"):
        _copy(repository_root / name, package / name)
    for name in ("train_six_models_server_3000.py", "requirements-server-training.txt"):
        _copy(HERE / name, package / name)
    for name in required_data:
        _copy(dataset / name, package / DATASET_NAME / name)

    (package / "README_SERVER.md").write_text(
        "# FALCON 10 m/s hybrid 160k — server training package\n\n"
        "This package contains a prepared 12-D train/validation/test split and the six-model fixed-3000 runner.\n\n"
        "```bash\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "python -m pip install --upgrade pip\n"
        "python -m pip install -r requirements-server-training.txt\n"
        "python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\"\n"
        "python train_six_models_server_3000.py --device cuda\n"
        "```\n\n"
        "Default batch size is 1024 for an approximately 11 GB NVIDIA GPU. If CUDA reports out of memory, restart with `--batch-size 512`; all completed epochs are retained in `latest_checkpoint.pt`.\n",
        encoding="utf-8",
    )

    files = sorted(path for path in package.rglob("*") if path.is_file())
    sums = "".join(f"{_sha256(path)}  {path.relative_to(package).as_posix()}\n" for path in files)
    (package / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(path for path in package.rglob("*") if path.is_file()):
            bundle.write(path, Path(PACKAGE_NAME) / path.relative_to(package))
    print(f"Package directory: {package}")
    print(f"Upload archive: {archive}")
    print(f"Archive bytes: {archive.stat().st_size}")


if __name__ == "__main__":
    main()
