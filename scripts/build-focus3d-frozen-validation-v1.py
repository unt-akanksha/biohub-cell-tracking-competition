from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT
    / ".biohub/research/public-kernel-audit-20260907/focus3d-physical-pp"
)
SOURCE_NOTEBOOK = SOURCE_DIR / "focus3d-nuclei-physical-pp-submit.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
SOURCE_NOTEBOOK_SHA256 = (
    "459cedbe2068fff6b4dfc57e20a9103a14bbc059baa4612c9f7693433a618de4"
)
SOURCE_METADATA_SHA256 = (
    "36462837cf1fbb21eb791422fb76500b8e8a76b869367d78bc5839b66c1ca784"
)
TARGET_ID = "biohub-focus3d-frozen-validation-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
FROZEN_STEMS = (
    "44b6_d754aa59",
    "44b6_7a302da0",
    "6bba_debd7bfa",
    "6bba_fc5f39dc",
)
PATCHED_METRICS_SHA256 = (
    "31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83"
)
PATCHED_DIVISION_METRICS_SHA256 = (
        "d1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def find_cell(notebook: dict, pattern: str) -> int:
    matches = [
        index
        for index, cell in enumerate(notebook["cells"])
        if pattern in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"FOCUS-3D source drift for {pattern!r}: {matches}")
    return matches[0]


ATTRIBUTION = """## Frozen FOCUS-3D detector validation

This is an evaluation-only derivative of
`qiweiyin/focus3d-nuclei-physical-pp-submit`. FOCUS-3D source is BSD-3-Clause
and its authors now explicitly publish the pretrained weights under
Apache-2.0. No public prediction is copied and no leaderboard result selects
any setting here.

The four complete movies, model checkpoint, inference settings, physical
linker, metric implementation, and pass/fail reporting are frozen before this
run. This notebook cannot submit: it writes neither a hidden-test prediction
nor `submission.csv`.
"""


WATCHDOG = r'''# Fail closed before Kaggle's 12-hour notebook limit.
import json as _fv_json
import os as _fv_os
from pathlib import Path as _FVPath
import threading as _fv_threading
import time as _fv_time

_FV_RUN_ID = "focus3d-frozen-validation-v1"
_FV_STARTED = _fv_time.time()
_FV_FINISHED = False
_FV_TERMINAL = _FVPath("/kaggle/working/focus3d_frozen_validation_terminal.json")

def _fv_write(status, **extra):
    payload = {
        "schema_version": 1,
        "run_id": _FV_RUN_ID,
        "status": status,
        "elapsed_seconds": _fv_time.time() - _FV_STARTED,
        "frozen_stems": [
            "44b6_d754aa59", "44b6_7a302da0",
            "6bba_debd7bfa", "6bba_fc5f39dc",
        ],
        "competition_test_data_read": False,
        "public_predictions_copied": False,
        "leaderboard_used_for_selection": False,
        "metric_hack_used": False,
        "submission_created": False,
        "authorized_for_submission": False,
        **extra,
    }
    _FV_TERMINAL.write_text(
        _fv_json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _fv_timeout():
    if not _FV_FINISHED:
        _fv_write("watchdog_timeout")
        _fv_os._exit(124)

_fv_timer = _fv_threading.Timer(10_800, _fv_timeout)
_fv_timer.daemon = True
_fv_timer.start()
_fv_write("running")
'''


INTEGRITY = f'''# Bind the patched scorer and exact two-GPU execution shape.
import hashlib as _fv_hashlib
import inspect as _fv_inspect
import biohub_tracking.metrics as _fv_metrics
import biohub_tracking.division_metrics as _fv_division_metrics

def _fv_sha256(path):
    digest = _fv_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_fv_metric_sha = _fv_sha256(Path(_fv_inspect.getfile(_fv_metrics)))
_fv_division_sha = _fv_sha256(Path(_fv_inspect.getfile(_fv_division_metrics)))
if _fv_metric_sha != "{PATCHED_METRICS_SHA256}":
    raise RuntimeError(f"Patched metrics source changed: {{_fv_metric_sha}}")
if _fv_division_sha != "{PATCHED_DIVISION_METRICS_SHA256}":
    raise RuntimeError(f"Patched division metrics source changed: {{_fv_division_sha}}")
_fv_gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(_fv_gpu_names) != 2 or any("T4" not in name for name in _fv_gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs required, saw {{_fv_gpu_names}}")
print({{"patched_metrics_sha256": _fv_metric_sha,
        "patched_division_metrics_sha256": _fv_division_sha,
        "gpu_names": _fv_gpu_names}})
'''


NO_SUBMISSION = r'''# Evaluation run: verify predictions exist without creating submission.csv.
_fv_prediction_paths = [Path(predict_dir) / f"{stem}.geff" for stem in valid_id]
_fv_missing = [str(path) for path in _fv_prediction_paths if not path.exists()]
if _fv_missing:
    raise RuntimeError(f"Missing frozen validation predictions: {_fv_missing}")
if Path("/kaggle/working/submission.csv").exists():
    raise RuntimeError("Evaluation-only run unexpectedly created submission.csv")
print(f"Verified {len(_fv_prediction_paths)} complete-movie prediction graphs")
'''


EVALUATION = r'''# Patched official complete-movie evaluation and immutable terminal.
metric_rows = []
per_movie = []
for sample_id in valid_id:
    truth_file = f"{KAGGLE_DIR}/train/{sample_id}.zarr"
    ds = open_dataset(
        truth_file, normalize=False, load_image=False, require_tracks=True
    )
    pred_graph = td.graph.IndexedRXGraph.from_geff(
        f"{predict_dir}/{sample_id}.geff"
    )[0]
    er = evaluate(pred_graph, ds.tracks, scale=ds.scale, max_distance=7.0)
    recall = node_recall(pred_graph, ds.tracks)
    meta = GeffMetadata.read(truth_file.replace(".zarr", ".geff"))
    row = per_sample_metrics(
        er=er,
        n_total=float(meta.extra["estimated_number_of_nodes"]),
        node_recall=recall,
    )
    row["stem"] = sample_id
    row["embryo"] = sample_id.split("_")[0]
    metric_rows.append(row)
    per_movie.append(row)

summary = summarise(metric_rows)
if len(metric_rows) != 4 or {row["stem"] for row in metric_rows} != set(valid_id):
    raise RuntimeError("Frozen complete-movie coverage changed")
by_embryo = {
    embryo: summarise([row for row in metric_rows if row["embryo"] == embryo])
    for embryo in ("44b6", "6bba")
}
result = {
    "complete_movie_count": len(metric_rows),
    "summary": summary,
    "by_embryo": by_embryo,
    "per_movie": per_movie,
    "focus3d_source_notebook_sha256":
        "459cedbe2068fff6b4dfc57e20a9103a14bbc059baa4612c9f7693433a618de4",
    "focus3d_code_license": "BSD-3-Clause",
    "focus3d_weights_license": "Apache-2.0",
    "standalone_gate_passed": bool(
        summary["score"] >= 0.93
        and min(row["adj_edge_jaccard"] for row in metric_rows) >= 0.85
        and all(by_embryo[embryo]["node_recall"] >= 0.95 for embryo in by_embryo)
    ),
}
_FV_FINISHED = True
_fv_timer.cancel()
_fv_write("completed", **result)
print(_fv_json.dumps(result, indent=2, sort_keys=True))
'''


def build_notebook() -> dict:
    if sha256_file(SOURCE_NOTEBOOK) != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("Audited FOCUS-3D notebook source changed")
    if sha256_file(SOURCE_METADATA) != SOURCE_METADATA_SHA256:
        raise RuntimeError("Audited FOCUS-3D notebook metadata changed")
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    setup_index = find_cell(notebook, 'MODE = "submit"')
    setup = "".join(notebook["cells"][setup_index]["source"])
    setup = setup.replace('MODE = "submit"', 'MODE = "local"', 1)
    old_stems = (
        'valid_id = ["44b6_0113de3b", "44b6_0b24845f", '
        '"6bba_05b6850b", "6bba_05db0fb1"]'
    )
    new_stems = f"valid_id = {list(FROZEN_STEMS)!r}"
    if setup.count(old_stems) != 1:
        raise RuntimeError("FOCUS-3D local validation selection changed")
    setup = setup.replace(old_stems, new_stems, 1)
    notebook["cells"][setup_index]["source"] = setup.splitlines(keepends=True)

    config_index = find_cell(notebook, "FOCUS3D_COMPILE = True")
    config = "".join(notebook["cells"][config_index]["source"])
    config = config.replace("FOCUS3D_COMPILE = True", "FOCUS3D_COMPILE = False", 1)
    notebook["cells"][config_index]["source"] = config.splitlines(keepends=True)

    submission_index = find_cell(notebook, "SUBMISSION_PATH = \"submission.csv\"")
    notebook["cells"][submission_index] = code_cell(NO_SUBMISSION)
    evaluation_index = find_cell(notebook, "#evalute")
    notebook["cells"][evaluation_index] = code_cell(EVALUATION)

    setup_index = find_cell(notebook, 'MODE = "local"')
    notebook["cells"][setup_index:setup_index] = [
        markdown_cell(ATTRIBUTION),
        code_cell(WATCHDOG),
    ]
    setup_index = find_cell(notebook, 'MODE = "local"')
    notebook["cells"][setup_index + 1 : setup_index + 1] = [code_cell(INTEGRITY)]

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["codex"] = {
        "run_id": "focus3d-frozen-validation-v1",
        "frozen_stems": list(FROZEN_STEMS),
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "metric_hack_used": False,
        "submission_command_included": False,
    }
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if TARGET_DIR.exists():
        if not args.replace:
            raise FileExistsError(f"Target already exists: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook()
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    source_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub FOCUS3D Frozen Validation v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    metadata.pop("id_no", None)
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
