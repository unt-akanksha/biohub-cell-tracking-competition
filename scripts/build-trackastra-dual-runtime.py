from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-trackastra-dual-runtime-v1"
TRACKASTRA_REPO = ROOT / ".biohub" / "cache" / "repos" / "trackastra"
MODEL_ROOT = ROOT / ".biohub" / "cache" / "models" / "trackastra-ctc-v0.3.0" / "ctc"
BASE_TRAINER = ROOT / "research" / "trackastra_graph" / "train_biohub_graph_transformer.py"
DUAL_TRAINER = ROOT / "research" / "trackastra_graph" / "train_dual_fold_synthetic.py"
SYNTHETIC_ADAPTER = ROOT / "research" / "synthetic_pretrain" / "data.py"
HYBRID_LINKER = ROOT / "research" / "trackastra_graph" / "hybrid_linker.py"
TRACKASTRA_FILES = (
    "trackastra/__init__.py",
    "trackastra/_version.py",
    "trackastra/model/__init__.py",
    "trackastra/model/model.py",
    "trackastra/model/model_parts.py",
    "trackastra/model/rope.py",
    "trackastra/utils/__init__.py",
    "trackastra/utils/utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checked_target() -> Path:
    staging_root = STAGING_ROOT.resolve()
    target = TARGET.resolve()
    if target.parent != staging_root or target.name != "biohub-trackastra-dual-runtime-v1":
        raise RuntimeError(f"unsafe runtime staging target: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    target = checked_target()
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} exists; pass --replace to rebuild")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    shutil.copy2(BASE_TRAINER, target / "trainer.py")
    shutil.copy2(DUAL_TRAINER, target / "dual_trainer.py")
    shutil.copy2(SYNTHETIC_ADAPTER, target / "synthetic_data.py")
    shutil.copy2(HYBRID_LINKER, target / "hybrid_linker.py")
    shutil.copy2(TRACKASTRA_REPO / "LICENSE", target / "TRACKASTRA_LICENSE")
    for relative in TRACKASTRA_FILES:
        source = TRACKASTRA_REPO / relative
        destination = target / "trackastra_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    pretrained = target / "ctc"
    pretrained.mkdir()
    for name in ("config.yaml", "train_config.yaml", "model.pt"):
        shutil.copy2(MODEL_ROOT / name, pretrained / name)

    files = {
        path.relative_to(target).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "purpose": "two-GPU reciprocal-fold synthetic Trackastra training; no submission path",
            "run_id": "trackastra-dual-fold-synthetic-v1",
            "trackastra": {
                "repository": "https://github.com/weigertlab/trackastra",
                "commit": "6a8ce94ee7c5a1f22c8eb77229ea5a0bc95a7b5b",
                "license": "BSD-3-Clause",
            },
            "pretrained_model": {
                "name": "ctc",
                "release": "v0.3.0",
                "model_sha256": files["ctc/model.pt"]["sha256"],
                "parameter_count": 27456880,
            },
            "integrity": {
                "opened_acceptance_labels_excluded": True,
                "synthetic_native_geometry_restored": True,
                "leaderboard_used_for_selection": False,
                "submission_created": False,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub Trackastra Dual Runtime v1",
            "id": "indarkarhana/biohub-trackastra-dual-runtime-v1",
            "licenses": [{"name": "BSD-3-Clause"}],
            "isPrivate": True,
        },
    )
    total_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    print(json.dumps({"target": str(target), "files": len(files) + 2, "bytes": total_bytes}))


if __name__ == "__main__":
    main()
