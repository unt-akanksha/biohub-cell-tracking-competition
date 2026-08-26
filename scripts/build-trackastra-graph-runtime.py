from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-trackastra-graph-runtime-v1"
TRACKASTRA_REPO = ROOT / ".biohub" / "cache" / "repos" / "trackastra"
MODEL_ROOT = ROOT / ".biohub" / "cache" / "models" / "trackastra-ctc-v0.3.0" / "ctc"
VALIDATION_ROOT = (
    ROOT
    / ".biohub"
    / "cache"
    / "notebook-outputs"
    / "evgendvorkin-biohub-0-927-lb-redownload"
    / "tracking_repo"
    / "predictions"
    / "unknown"
    / "unet_transformer_val"
    / "split_0"
)
TRAINER = ROOT / "research" / "trackastra_graph" / "train_biohub_graph_transformer.py"
SYNTHETIC_ADAPTER = ROOT / "research" / "synthetic_pretrain" / "data.py"
HYBRID_LINKER = ROOT / "research" / "trackastra_graph" / "hybrid_linker.py"
SUBMISSION_RERANKER = ROOT / "research" / "trackastra_graph" / "rerank_submission.py"
POSTHOC_VALIDATOR = ROOT / "research" / "trackastra_graph" / "validate_model_posthoc.py"
PUBLIC_VALIDATION_MATERIALIZER = (
    ROOT / "research" / "trackastra_graph" / "materialize_public_validation.py"
)
PUBLIC_NOTEBOOK = (
    ROOT
    / ".biohub"
    / "cache"
    / "notebooks"
    / "evgendvorkin"
    / "biohub-0-927-lb"
    / "biohub-0-927-lb.ipynb"
)
PUBLIC_NOTEBOOK_SHA256 = "08507f9123d9f40e185d0db8eda3dd21cb405e50febb720827c2e655f68d5ec1"
VALIDATION_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
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
    root = STAGING_ROOT.resolve()
    target = TARGET.resolve()
    if target.parent != root or target.name != "biohub-trackastra-graph-runtime-v1":
        raise RuntimeError(f"Unsafe staging target: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    target = checked_target()
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} already exists; pass --replace to rebuild")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    shutil.copy2(TRAINER, target / "trainer.py")
    shutil.copy2(SYNTHETIC_ADAPTER, target / "synthetic_data.py")
    shutil.copy2(HYBRID_LINKER, target / "hybrid_linker.py")
    shutil.copy2(SUBMISSION_RERANKER, target / "rerank_submission.py")
    shutil.copy2(POSTHOC_VALIDATOR, target / "validate_model_posthoc.py")
    shutil.copy2(
        PUBLIC_VALIDATION_MATERIALIZER,
        target / "materialize_public_validation.py",
    )
    if sha256_file(PUBLIC_NOTEBOOK) != PUBLIC_NOTEBOOK_SHA256:
        raise RuntimeError("Audited public comparator notebook source changed")
    public_notebook = json.loads(PUBLIC_NOTEBOOK.read_text(encoding="utf-8"))
    public_config_source = "".join(public_notebook["cells"][7]["source"])
    public_postprocess_cell = "".join(public_notebook["cells"][13]["source"])
    public_stop_marker = "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()"
    if public_stop_marker not in public_postprocess_cell:
        raise RuntimeError("Audited public postprocessor boundary changed")
    public_postprocess_source = public_postprocess_cell.split(
        public_stop_marker, maxsplit=1
    )[0]
    (target / "public_config_source.py").write_text(
        public_config_source, encoding="utf-8"
    )
    (target / "public_postprocess_source.py").write_text(
        public_postprocess_source, encoding="utf-8"
    )
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

    validator = target / "validator_raw"
    validator.mkdir()
    for stem in VALIDATION_STEMS:
        source_graph = VALIDATION_ROOT / f"{stem}.geff"
        required = (
            source_graph / "zarr.json",
            source_graph / "nodes" / "zarr.json",
            source_graph / "edges" / "zarr.json",
            source_graph / "nodes" / "ids" / "c" / "0",
            source_graph / "edges" / "ids" / "c" / "0" / "0",
        )
        missing = [path for path in required if not path.is_file()]
        file_count = sum(1 for path in source_graph.rglob("*") if path.is_file())
        if missing or file_count != 33:
            raise RuntimeError(
                f"Incomplete validator graph {stem}: missing={missing}, files={file_count}"
            )
        shutil.copytree(source_graph, validator / f"{stem}.geff")

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
            "purpose": "Biohub-native Trackastra association fine-tuning; not a public output replica",
            "trackastra": {
                "repository": "https://github.com/weigertlab/trackastra",
                "commit": "6a8ce94ee7c5a1f22c8eb77229ea5a0bc95a7b5b",
                "license": "BSD-3-Clause",
                "local_import_patch": [
                    "generated trackastra/_version.py for source checkout",
                    "removed eager model_api import from trackastra/model/__init__.py",
                ],
            },
            "pretrained_model": {
                "name": "ctc",
                "release": "v0.3.0",
                "source": "official Trackastra pretrained release",
                "model_sha256": files["ctc/model.pt"]["sha256"],
            },
            "validation": {
                "source": "frozen raw detector predictions from public 0.927 pipeline",
                "stems": list(VALIDATION_STEMS),
                "selection_signal": "complete-movie clean heldout metric only",
            },
            "public_validation_postprocess": {
                "role": "frozen comparator materialization only",
                "source_notebook": "evgendvorkin/biohub-0-927-lb",
                "source_notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
                "config_cell_sha256": files["public_config_source.py"]["sha256"],
                "postprocess_prefix_sha256": files[
                    "public_postprocess_source.py"
                ]["sha256"],
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub Trackastra Graph Runtime v1",
            "id": "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "licenses": [{"name": "BSD-3-Clause"}],
            "isPrivate": True,
        },
    )
    total_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    print(json.dumps({"target": str(target), "files": len(files) + 2, "bytes": total_bytes}))


if __name__ == "__main__":
    main()
