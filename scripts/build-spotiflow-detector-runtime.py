from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".biohub" / "staging" / "biohub-spotiflow-detector-runtime-v1"
MODEL_CACHE = ROOT / ".biohub" / "cache" / "models" / "spotiflow-0.6.0"
WHEEL_CACHE = ROOT / ".biohub" / "cache" / "wheels" / "spotiflow-py312-linux"
REPO = ROOT / ".biohub" / "cache" / "repos" / "spotiflow"
EXPECTED_ARCHIVE_MD5 = {
    "synth_3d": "a031f1284590886fbae37dc583c0270d",
    "smfish_3d": "c5ab30ba3b9ccb07b4c34442d1b5b615",
}
MODEL_FILES = ("best.pt", "config.yaml", "thresholds.yaml", "train_config.yaml")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    target = TARGET.resolve()
    staging_root = (ROOT / ".biohub" / "staging").resolve()
    if staging_root not in target.parents or target.name != "biohub-spotiflow-detector-runtime-v1":
        raise RuntimeError(f"Refusing to replace unexpected target: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    models_manifest = {}
    for model_name, expected_md5 in EXPECTED_ARCHIVE_MD5.items():
        archive = MODEL_CACHE / f"{model_name}.zip"
        actual_md5 = file_hash(archive, "md5")
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"{model_name} archive MD5 mismatch: {actual_md5} != {expected_md5}"
            )
        source_dir = MODEL_CACHE / model_name
        destination = target / "models" / model_name
        destination.mkdir(parents=True)
        copied = {}
        for name in MODEL_FILES:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / name)
            copied[name] = {
                "bytes": source.stat().st_size,
                "sha256": file_hash(source),
            }
        models_manifest[model_name] = {
            "official_archive_md5": actual_md5,
            "files": copied,
        }

    wheels_dir = target / "wheels"
    wheels_dir.mkdir()
    wheels_manifest = {}
    for source in sorted(WHEEL_CACHE.glob("*.whl")):
        shutil.copy2(source, wheels_dir / source.name)
        wheels_manifest[source.name] = {
            "bytes": source.stat().st_size,
            "sha256": file_hash(source),
        }
    spotiflow_wheels = [name for name in wheels_manifest if name.startswith("spotiflow-")]
    if len(spotiflow_wheels) != 1:
        raise RuntimeError(f"Expected one Spotiflow wheel, found {spotiflow_wheels}")

    script_sources = {
        "evaluate_pretrained_detector.py": (
            ROOT / "research" / "spotiflow_biohub" / "evaluate_pretrained_detector.py"
        ),
        "density_calibration.py": ROOT / "research" / "density_calibration.py",
        "train_synthetic_detector.py": (
            ROOT / "research" / "spotiflow_biohub" / "train_synthetic_detector.py"
        ),
        "synthetic_data.py": ROOT / "research" / "synthetic_pretrain" / "data.py",
    }
    for name, source in script_sources.items():
        shutil.copy2(source, target / name)
    license_path = next((path for path in (REPO / "LICENSE", REPO / "LICENSE.txt") if path.is_file()), None)
    if license_path is None:
        raise FileNotFoundError("Spotiflow license file not found")
    shutil.copy2(license_path, target / "SPOTIFLOW_LICENSE")

    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "dataset_slug": "indarkarhana/biohub-spotiflow-detector-runtime-v1",
        "purpose": "official Spotiflow 3D detector acceptance and corrected synthetic fine-tuning",
        "spotiflow_repository": {
            "url": "https://github.com/weigertlab/spotiflow",
            "commit": commit,
            "license": "BSD-3-Clause",
            "wheel_version": "0.6.2",
        },
        "spotiflow_models": models_manifest,
        "wheels": wheels_manifest,
        "scripts": {name: file_hash(target / name) for name in script_sources},
    }
    (target / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Biohub Spotiflow Detector Runtime v1",
                "id": "indarkarhana/biohub-spotiflow-detector-runtime-v1",
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
