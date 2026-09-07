#!/usr/bin/env python
"""Build a two-GPU peak-ranking plus audited-linker submission candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / ".biohub"
    / "research"
    / "public-kernel-audit-20260907"
    / "redoctopusk-biohub-948tta2"
)
SOURCE_NOTEBOOK = SOURCE_ROOT / "biohub-948tta2.ipynb"
SOURCE_METADATA = SOURCE_ROOT / "kernel-metadata.json"
SOURCE_NOTEBOOK_SHA256 = (
    "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189"
)
SOURCE_METADATA_SHA256 = (
    "13c98287dd9228c0d8f9abd764e01b3484837ddc4cc0f4dee6e30f5df06086bb"
)
SOURCE_KERNEL_REF = "redoctopusk/biohub-948tta2"
RUNTIME_REF = "indarkarhana/biohub-peak-rank-validation-runtime-v1"
TARGET_ID = "biohub-peak-rank-tracking-candidate-v1"
CANDIDATE_RUN_ID = "peak-rank-tracking-candidate-v1"
TARGET_TITLE = "Biohub Peak Rank Tracking Candidate v1"
TARGET = ROOT / "kaggle" / TARGET_ID
NOTEBOOK = TARGET / f"{TARGET_ID}.ipynb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_exact(source: str, old: str, new: str, *, count: int = 1) -> str:
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(
            f"public source drift for {old!r}: expected {count}, saw {actual}"
        )
    return source.replace(old, new, count)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


WATCHDOG = r"""import atexit as _pr_atexit
import json as _pr_json
import os as _pr_os
import threading as _pr_threading
import time as _pr_time
from pathlib import Path as _PR_Path

_PR_RUN_ID = "peak-rank-tracking-candidate-v1"
_PR_STARTED = _pr_time.monotonic()
_PR_FINISHED = False
_PR_TERMINAL = _PR_Path("/kaggle/working/launcher_terminal.json")

def _pr_write_terminal(status, error=None):
    submission = _PR_Path("/kaggle/working/submission.csv")
    evidence = _PR_Path("/kaggle/working/candidate_evidence.json")
    payload = {
        "schema_version": 1,
        "run_id": _PR_RUN_ID,
        "status": status,
        "elapsed_seconds": round(_pr_time.monotonic() - _PR_STARTED, 3),
        "declared_budget_seconds": 43200,
        "hard_stop_seconds": 41400,
        "submission_exists": submission.is_file(),
        "evidence_exists": evidence.is_file(),
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    if error is not None:
        payload["error"] = str(error)
    temporary = _PR_TERMINAL.with_suffix(".partial")
    temporary.write_text(_pr_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(_PR_TERMINAL)

def _pr_abort():
    if not _PR_FINISHED:
        _pr_write_terminal("aborted")

def _pr_expire():
    _pr_write_terminal("budget_expired")
    _pr_os._exit(124)

_pr_atexit.register(_pr_abort)
_PR_TIMER = _pr_threading.Timer(41400, _pr_expire)
_PR_TIMER.daemon = True
_PR_TIMER.start()
print("Peak-ranking candidate watchdog armed for 41,400 seconds.")
"""


ATTRIBUTION = """## Independent detector plus attributed tracking backbone

This candidate does not copy a public prediction. Its node generator is the
project-authored 38.4M-parameter temporal peak-ranking detector, trained on
synthetic complete labels and positive-only real supervision. The association,
ILP, DeepCenter, and graph-finishing code retains attribution to the audited
RedOctopusk/Pilkwang public lineage. Its secondary edge-feature TTA reuses the
detector TTA passes already computed by that lineage; it is attributed here as
linker-side inference, not an independent model.
The notebook title's advertised `0.948` is not treated as reproduced evidence
and does not select this candidate.

Execution requires the detector to have passed its sealed training audit and
the separate two-GPU complete-movie clean validation. Submission remains an
external decision after the materialized four-movie graphs beat the frozen
clean control under the locally pinned patched official scorer without a
movie-level regression. The public notebook's local metric reimplementation is
retained only as a diagnostic and cannot authorize submission.
"""


RUNTIME_SETUP = r"""# Locate and verify the clean-promoted private detector runtime.
import hashlib as _pr_hashlib
import json as _pr_json
import torch as _pr_torch

_PR_MANIFEST_SHA256 = "__MANIFEST_SHA256__"
def _pr_sha256(path):
    digest = _pr_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_pr_matches = [
    path for path in Path("/kaggle/input").rglob("SOURCE_MANIFEST.json")
    if _pr_sha256(path) == _PR_MANIFEST_SHA256
]
if len(_pr_matches) != 1:
    raise RuntimeError(f"Expected one promoted peak runtime, saw {_pr_matches}")
_PR_ROOT = _pr_matches[0].parent
_PR_MANIFEST = _pr_json.loads(_pr_matches[0].read_text(encoding="utf-8"))
for _name, _record in _PR_MANIFEST["files"].items():
    if _pr_sha256(_PR_ROOT / _name) != _record["sha256"]:
        raise RuntimeError(f"Peak runtime changed: {_name}")
_PR_VALIDATION = _pr_json.loads((_PR_ROOT / "clean_validation.json").read_text(encoding="utf-8"))
if not (
    _PR_MANIFEST.get("training_audit_passed") is True
    and _PR_MANIFEST.get("clean_validation_promotion_passed") is True
    and _PR_MANIFEST.get("clean_validation_sha256") == _pr_sha256(_PR_ROOT / "clean_validation.json")
    and _PR_VALIDATION.get("selection_passed") is True
    and _PR_VALIDATION.get("acceptance_opened") is True
    and _PR_VALIDATION.get("promotion_passed") is True
    and _PR_VALIDATION.get("selected_tta_mode") in {"none", "zflip2", "rot4", "d4"}
    and _PR_MANIFEST.get("selected_peak_tta_mode") == _PR_VALIDATION.get("selected_tta_mode")
    and _PR_MANIFEST.get("selected_peak_tta_views") == _PR_VALIDATION.get("selected_tta_views")
    and _PR_VALIDATION.get("competition_submission_performed") is False
    and _PR_VALIDATION.get("provenance", {}).get("checkpoint_sha256")
        == _PR_MANIFEST.get("checkpoint_sha256")
):
    raise RuntimeError("Peak-ranking clean promotion evidence is ineligible")
_PR_TTA_MODE = _PR_VALIDATION["selected_tta_mode"]
if not _pr_torch.cuda.is_available() or _pr_torch.cuda.device_count() != 2:
    raise RuntimeError("Peak-ranking production requires exactly two Kaggle GPUs")
print(_pr_json.dumps({
    "peak_runtime": str(_PR_ROOT),
    "checkpoint_sha256": _PR_MANIFEST["checkpoint_sha256"],
    "runtime_manifest_sha256": _PR_MANIFEST_SHA256,
    "clean_validation_sha256": _PR_MANIFEST["clean_validation_sha256"],
    "selected_peak_tta_mode": _PR_TTA_MODE,
    "gpu_count": _pr_torch.cuda.device_count(),
}, indent=2, sort_keys=True))
"""


CANDIDATE_EVIDENCE = r"""# Hash-bind the complete two-worker output for external promotion.
_pr_manifest_root = Path("/kaggle/working/peak_worker_manifests")
_pr_worker_paths = sorted(_pr_manifest_root.glob("worker-*.json"))
if len(_pr_worker_paths) != 2:
    raise RuntimeError(f"Expected two peak worker manifests, saw {_pr_worker_paths}")
_pr_workers = [_pr_json.loads(path.read_text(encoding="utf-8")) for path in _pr_worker_paths]
if not all(
    row.get("run_id") == "peak-rank-official-linker-production-v1"
    and row.get("worker_count") == 2
    and row.get("checkpoint_sha256") == _PR_MANIFEST["checkpoint_sha256"]
    and row.get("parameter_count") == _PR_MANIFEST["parameter_count"]
    and row.get("ensemble_size") == _PR_MANIFEST["ensemble_size"]
    and row.get("peak_tta_mode") == _PR_TTA_MODE
    and row.get("peak_tta_views") == _PR_MANIFEST["selected_peak_tta_views"]
    and row.get("association", {}).get("edge_feature_tta") is True
    and row.get("association", {}).get("secondary_link_mode") == "low_margin_consensus"
    and row.get("association", {}).get("bidirectional_edge_weight") == 0.15
    and row.get("association", {}).get("association_coordinate_mode") == "subvoxel"
    and row.get("input_partition") == "test"
    and row.get("competition_train_labels_read") is False
    and row.get("competition_test_labels_read") is False
    and row.get("public_predictions_copied") is False
    and row.get("public_leaderboard_used_for_selection") is False
    and row.get("submission_created") is False
    for row in _pr_workers
):
    raise RuntimeError("Peak worker evidence is ineligible")
_pr_observed = {
    movie["dataset"] for row in _pr_workers for movie in row["movies"]
}
if _pr_observed != set(test_stems) or sum(len(row["movies"]) for row in _pr_workers) != len(test_stems):
    raise RuntimeError("Peak worker movie coverage is incomplete or duplicated")
if not validator_summary_rows or len(validator_sample_rows) < 4:
    raise RuntimeError("Complete-movie candidate validator evidence is missing")
_pr_official_validator = Path("/kaggle/working/official_validator_candidate.csv")
if not _pr_official_validator.is_file():
    raise FileNotFoundError(_pr_official_validator)
_pr_submission = Path("/kaggle/working/submission.csv")
if not _pr_submission.is_file():
    raise FileNotFoundError(_pr_submission)
_pr_evidence = {
    "schema_version": 1,
    "run_id": "peak-rank-tracking-candidate-v1",
    "status": "completed_pending_external_promotion_gate",
    "target_public_score": 0.945,
    "source_public_lineage_attributed": True,
    "source_public_kernel_ref": "redoctopusk/biohub-948tta2",
    "source_public_notebook_sha256": "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189",
    "secondary_edge_feature_tta": True,
    "subvoxel_association_coordinates": True,
    "dual_association_models_verified": True,
    "source_advertised_score_used_as_evidence": False,
    "public_predictions_copied": False,
    "checkpoint_sha256": _PR_MANIFEST["checkpoint_sha256"],
    "parameter_count": _PR_MANIFEST["parameter_count"],
    "ensemble_size": _PR_MANIFEST["ensemble_size"],
    "clean_validation_sha256": _PR_MANIFEST["clean_validation_sha256"],
    "selected_peak_tta_mode": _PR_TTA_MODE,
    "selected_peak_tta_views": _PR_MANIFEST["selected_peak_tta_views"],
    "max_worker_elapsed_seconds": max(float(row["worker_elapsed_seconds"]) for row in _pr_workers),
    "max_projected_worker_seconds": max(float(row["projected_worker_seconds"]) for row in _pr_workers),
    "worker_budget_seconds": min(float(row["worker_budget_seconds"]) for row in _pr_workers),
    "worker_count": 2,
    "complete_test_movie_count": len(_pr_observed),
    "complete_validator_movie_count": len(validator_sample_rows),
    "official_metric_status": "pending_external_patched_official_scoring",
    "official_scorer_lock_sha256": "1db65dee620059f19bf16633aa54a9f3379eb5d5bdff148a4037b949393b7a9c",
    "official_validator_candidate_sha256": _pr_sha256(_pr_official_validator),
    "validator_proxy_score": float(validator_summary_rows[-1]["proxy_score"]),
    "validator_adjusted_edge_jaccard": float(validator_summary_rows[-1]["adjusted_edge_jaccard"]),
    "validator_division_jaccard": float(validator_summary_rows[-1]["division_jaccard"]),
    "submission_sha256": _pr_sha256(_pr_submission),
    "competition_submission_performed": False,
    "authorized_for_submission": False,
}
Path("/kaggle/working/candidate_evidence.json").write_text(
    _pr_json.dumps(_pr_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
_PR_FINISHED = True
_PR_TIMER.cancel()
_pr_write_terminal("completed")
print(_pr_json.dumps(_pr_evidence, indent=2, sort_keys=True))
"""


def verify_promoted_runtime(runtime_root: Path) -> str:
    manifest_path = runtime_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = runtime_root / "clean_validation.json"
    if not (
        manifest.get("training_audit_passed") is True
        and manifest.get("clean_validation_promotion_passed") is True
        and validation.is_file()
        and manifest.get("clean_validation_sha256") == sha256_file(validation)
        and manifest.get("checkpoint_sha256")
        == sha256_file(runtime_root / "peak_rank_detector.pt")
        and manifest.get("selected_peak_tta_mode") in {"none", "zflip2", "rot4", "d4"}
        and manifest.get("selected_peak_tta_views") in {1, 4, 8}
    ):
        raise RuntimeError("local peak runtime is not clean-promoted")
    return sha256_file(manifest_path)


def build_notebook(manifest_sha256: str) -> dict:
    if sha256_file(SOURCE_NOTEBOOK) != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("audited public notebook source changed")
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"][3] = markdown_cell(ATTRIBUTION)
    inference_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "Fail fast instead of silently running volumetric inference on CPU"
        in "".join(cell.get("source", []))
    )
    inference = "".join(notebook["cells"][inference_index]["source"])
    inference = replace_exact(
        inference,
        '    "scripts/predict_unet_transformer.py",\n',
        '    str(_PR_ROOT / "predict_with_official_linker.py"),\n'
        '    "--runtime-root",\n'
        "    str(_PR_ROOT),\n"
        '    "--peak-tta-mode",\n'
        "    _PR_TTA_MODE,\n"
        '    "--official-predictor",\n'
        '    str(REPO_DIR / "scripts/predict_unet_transformer.py"),\n',
    )
    inference = replace_exact(
        inference,
        "    for shard_dir in shard_dirs:\n        _shutil.rmtree(shard_dir.parent)\n",
        "    _manifest_root = Path('/kaggle/working/peak_worker_manifests')\n"
        "    _manifest_root.mkdir(parents=True, exist_ok=False)\n"
        "    for _manifest_index, shard_dir in enumerate(shard_dirs):\n"
        "        _manifest_source = shard_dir / 'worker_manifest.json'\n"
        "        if not _manifest_source.is_file():\n"
        "            raise FileNotFoundError(_manifest_source)\n"
        "        _shutil.copy2(_manifest_source, _manifest_root / f'worker-{_manifest_index}.json')\n"
        "        _shutil.rmtree(shard_dir.parent)\n",
    )
    notebook["cells"][inference_index]["source"] = inference.splitlines(keepends=True)
    validator_inference_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "predict_val_cmd = [" in "".join(cell.get("source", []))
    )
    validator_inference = "".join(
        notebook["cells"][validator_inference_index]["source"]
    )
    validator_inference = replace_exact(
        validator_inference,
        '        sys.executable, "scripts/predict_unet_transformer.py",\n',
        '        sys.executable, str(_PR_ROOT / "predict_with_official_linker.py"),\n'
        '        "--runtime-root", str(_PR_ROOT),\n'
        '        "--peak-tta-mode", _PR_TTA_MODE,\n'
        '        "--official-predictor", str(REPO_DIR / "scripts/predict_unet_transformer.py"),\n',
    )
    notebook["cells"][validator_inference_index]["source"] = (
        validator_inference.splitlines(keepends=True)
    )
    validator_scoring_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "def score_sample(" in "".join(cell.get("source", []))
    )
    validator_scoring = "".join(notebook["cells"][validator_scoring_index]["source"])
    validator_scoring = replace_exact(
        validator_scoring,
        "validator_sample_rows: list[dict[str, object]] = []\n"
        "validator_summary_rows: list[dict[str, object]] = []\n",
        "validator_sample_rows: list[dict[str, object]] = []\n"
        "validator_summary_rows: list[dict[str, object]] = []\n"
        "_pr_official_validator_rows: list[dict[str, object]] = []\n",
    )
    validator_scoring = replace_exact(
        validator_scoring,
        "        pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)\n"
        '        pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]\n',
        "        pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)\n"
        '        pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]\n'
        "        for _pr_node_id in sorted(processed_nodes):\n"
        "            _pr_node = processed_nodes[_pr_node_id]\n"
        "            _pr_official_validator_rows.append({\n"
        '                "dataset": stem, "row_type": "node",\n'
        '                "node_id": int(_pr_node["node_id"]), "t": int(_pr_node["t"]),\n'
        '                "z": max(0, int(round(float(_pr_node["z"])))),\n'
        '                "y": max(0, int(round(float(_pr_node["y"])))),\n'
        '                "x": max(0, int(round(float(_pr_node["x"])))),\n'
        '                "source_id": -1, "target_id": -1,\n'
        "            })\n"
        '        for _pr_edge in sorted(processed_edges, key=lambda e: (int(e["source_id"]), int(e["target_id"]))):\n'
        "            _pr_official_validator_rows.append({\n"
        '                "dataset": stem, "row_type": "edge",\n'
        '                "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,\n'
        '                "source_id": int(_pr_edge["source_id"]),\n'
        '                "target_id": int(_pr_edge["target_id"]),\n'
        "            })\n",
    )
    validator_scoring = replace_exact(
        validator_scoring,
        "        validator_sample_rows.append(row)\n\n    if rows_this_config:\n",
        "        validator_sample_rows.append(row)\n\n"
        "    _pr_expected_validator_stems = {\n"
        '        "44b6_12dfb391", "44b6_267148e4",\n'
        '        "6bba_062c8d37", "6bba_07e24132",\n'
        "    }\n"
        '    _pr_observed_validator_stems = {str(row["dataset"]) for row in _pr_official_validator_rows}\n'
        "    if _pr_observed_validator_stems != _pr_expected_validator_stems:\n"
        "        raise RuntimeError(\n"
        '            f"Official validator coverage mismatch: {_pr_observed_validator_stems}"\n'
        "        )\n"
        '    _pr_official_validator_path = WORKING_DIR / "official_validator_candidate.csv"\n'
        "    _pr_official_columns = [\n"
        '        "id", "dataset", "row_type", "node_id", "t",\n'
        '        "z", "y", "x", "source_id", "target_id",\n'
        "    ]\n"
        '    with _pr_official_validator_path.open("w", newline="") as _pr_stream:\n'
        "        _pr_writer = csv.DictWriter(_pr_stream, fieldnames=_pr_official_columns)\n"
        "        _pr_writer.writeheader()\n"
        "        for _pr_row_id, _pr_row in enumerate(_pr_official_validator_rows):\n"
        '            _pr_writer.writerow({"id": _pr_row_id, **_pr_row})\n'
        "    print(\n"
        '        f"Materialized {len(_pr_official_validator_rows)} official-score rows "\n'
        '        f"to {_pr_official_validator_path}"\n'
        "    )\n\n"
        "    if rows_this_config:\n",
    )
    notebook["cells"][validator_scoring_index]["source"] = validator_scoring.splitlines(
        keepends=True
    )
    watchdog = WATCHDOG.replace("peak-rank-tracking-candidate-v1", CANDIDATE_RUN_ID)
    evidence = CANDIDATE_EVIDENCE.replace(
        "peak-rank-tracking-candidate-v1", CANDIDATE_RUN_ID
    )
    notebook["cells"][inference_index:inference_index] = [
        code_cell(watchdog),
        code_cell(RUNTIME_SETUP.replace("__MANIFEST_SHA256__", manifest_sha256)),
    ]
    notebook["cells"].append(code_cell(evidence))
    notebook["metadata"]["codex"] = {
        "status": "candidate_requires_external_patched_official_promotion",
        "public_prediction_copied": False,
        "target_public_score": 0.945,
        "source_public_kernel_ref": SOURCE_KERNEL_REF,
        "source_public_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "secondary_edge_feature_tta": True,
        "runtime_manifest_sha256": manifest_sha256,
        "candidate_run_id": CANDIDATE_RUN_ID,
        "public_validator_proxy_can_promote": False,
        "official_scorer_lock_sha256": (
            "1db65dee620059f19bf16633aa54a9f3379eb5d5bdff148a4037b949393b7a9c"
        ),
    }
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if sha256_file(SOURCE_METADATA) != SOURCE_METADATA_SHA256:
        raise RuntimeError("audited public metadata changed")
    manifest_sha256 = verify_promoted_runtime(args.runtime_root)
    if TARGET.exists():
        if not args.replace:
            raise FileExistsError(TARGET)
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    NOTEBOOK.write_text(
        json.dumps(
            build_notebook(manifest_sha256), ensure_ascii=True, separators=(",", ":")
        ),
        encoding="ascii",
    )
    source_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": TARGET_TITLE,
        "code_file": NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [*source_metadata["dataset_sources"], RUNTIME_REF],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
