from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".biohub" / "staging" / "biohub-spotiflow-pu-runtime-v1"
MODEL_CACHE = ROOT / ".biohub" / "cache" / "models" / "spotiflow-0.6.0"
WHEEL_CACHE = ROOT / ".biohub" / "cache" / "wheels" / "spotiflow-py312-linux"
REPO = ROOT / ".biohub" / "cache" / "repos" / "spotiflow"
EXPECTED_SMISH_ARCHIVE_MD5 = "c5ab30ba3b9ccb07b4c34442d1b5b615"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    target = TARGET.resolve()
    staging = (ROOT / ".biohub" / "staging").resolve()
    if staging not in target.parents or target.name != "biohub-spotiflow-pu-runtime-v1":
        raise RuntimeError(f"refusing to replace unexpected path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    archive = MODEL_CACHE / "smfish_3d.zip"
    if file_hash(archive, "md5") != EXPECTED_SMISH_ARCHIVE_MD5:
        raise RuntimeError("official smfish_3d archive MD5 mismatch")
    model_source = MODEL_CACHE / "smfish_3d"
    model_target = target / "models" / "smfish_3d"
    model_target.mkdir(parents=True)
    model_files = {}
    for name in ("best.pt", "config.yaml", "thresholds.yaml", "train_config.yaml"):
        source = model_source / name
        shutil.copy2(source, model_target / name)
        model_files[name] = {
            "bytes": source.stat().st_size,
            "sha256": file_hash(source),
        }

    wheels_target = target / "wheels"
    wheels_target.mkdir()
    wheels = {}
    for source in sorted(WHEEL_CACHE.glob("*.whl")):
        shutil.copy2(source, wheels_target / source.name)
        wheels[source.name] = {
            "bytes": source.stat().st_size,
            "sha256": file_hash(source),
        }
    if len([name for name in wheels if name.startswith("spotiflow-")]) != 1:
        raise RuntimeError("runtime must contain exactly one Spotiflow wheel")

    scripts = {
        "pu_targets.py": ROOT / "research" / "spotiflow_biohub" / "pu_targets.py",
        "public_teacher.py": ROOT
        / "research"
        / "spotiflow_biohub"
        / "public_teacher.py",
        "train_pu_detector.py": ROOT
        / "research"
        / "spotiflow_biohub"
        / "train_pu_detector.py",
        "evaluate_pu_detector.py": ROOT
        / "research"
        / "spotiflow_biohub"
        / "evaluate_pu_detector.py",
        "evaluate_pretrained_detector.py": ROOT
        / "research"
        / "spotiflow_biohub"
        / "evaluate_pretrained_detector.py",
        "density_calibration.py": ROOT / "research" / "density_calibration.py",
    }
    for name, source in scripts.items():
        shutil.copy2(source, target / name)
    license_path = next(
        (path for path in (REPO / "LICENSE", REPO / "LICENSE.txt") if path.is_file()),
        None,
    )
    if license_path is None:
        raise FileNotFoundError("Spotiflow license not found")
    shutil.copy2(license_path, target / "SPOTIFLOW_LICENSE")
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "dataset_slug": "indarkarhana/biohub-spotiflow-pu-runtime-v1",
        "purpose": "positive-unlabeled Biohub adaptation and clean detector validation",
        "spotiflow": {
            "repository": "https://github.com/weigertlab/spotiflow",
            "commit": commit,
            "license": "BSD-3-Clause",
            "wheel_version": "0.6.2",
            "official_archive_md5": EXPECTED_SMISH_ARCHIVE_MD5,
            "model_files": model_files,
        },
        "wheels": wheels,
        "scripts": {name: file_hash(target / name) for name in scripts},
        "public_teacher_design": {
            "model": "two frozen TemporalUNet3D detector seeds",
            "weights_attached_separately": True,
            "public_code_copied_into_candidate_output": False,
        },
    }
    (target / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Biohub Spotiflow PU Runtime v1",
                "id": "indarkarhana/biohub-spotiflow-pu-runtime-v1",
                "licenses": [{"name": "BSD-3-Clause"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(target)
    print("manifest_sha256", file_hash(target / "SOURCE_MANIFEST.json"))


if __name__ == "__main__":
    main()
