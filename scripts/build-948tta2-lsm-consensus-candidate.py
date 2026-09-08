#!/usr/bin/env python
"""Build a clean 0.948-TTA2 control plus owned LSM coordinate consensus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT
    / ".biohub/research/public-refresh-20260908-r1"
    / "redoctopusk__biohub-948tta2"
)
SOURCE_NOTEBOOK = SOURCE_DIR / "biohub-948tta2.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
SOURCE_NOTEBOOK_SHA256 = (
    "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189"
)
SOURCE_METADATA_SHA256 = (
    "13c98287dd9228c0d8f9abd764e01b3484837ddc4cc0f4dee6e30f5df06086bb"
)
TARGET_ID = "biohub-948tta2-lsm-consensus-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = "indarkarhana/biohub-lsm-fm-public-node-runtime-v2"
FEATURE24_KERNEL = "indarkarhana/biohub-lsm-fm-pu-adaptation-v2"
FEATURE36_KERNEL = "indarkarhana/biohub-lsm-fm-image-text-pu-adaptation-v1"


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


def replace_exact(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"clean public control drifted for {old!r}: expected {count}, saw {actual}"
        )
    return text.replace(old, new, count)


ATTRIBUTION = """## Project candidate: clean 0.948-TTA2 plus LSM coordinate consensus

The detector, dual-seed association ensemble, edge-feature TTA, harmonic
bidirectional fusion, DeepCenter veto, graph reconstruction, and safe-division
stage retain their attribution to `redoctopusk/biohub-948tta2` and its listed
Pilkwang inputs. Its advertised leaderboard score is not validation evidence,
and no public prediction is copied.

The only candidate change is project-authored and topology preserving. Two
independently trained Biohub LSM-FM heatmap networks propose a local coordinate
refinement for each existing node. A node moves only when the two networks
produce exactly the same final integer `(z, y, x)` coordinate under one frozen
radius-2, squared-probability, 0.25-blend rule. Otherwise the public coordinate
is retained. The rule cannot add or delete a node or edge, change a node ID or
time, read an estimated node count, or create an out-of-bounds coordinate.

The notebook scores the untouched control and candidate on the same complete
held-out movies. Submission eligibility requires a strict patched-metric proxy
gain, no per-movie adjusted-edge regression, no division regression, at least
one changed production coordinate, and all graph-integrity assertions.
"""


WATCHDOG = r'''# Project runtime and fail-closed terminal receipt.
import atexit as _lc_atexit
import hashlib as _lc_hashlib
import json as _lc_json
import os as _lc_os
import threading as _lc_threading
import time as _lc_time
from pathlib import Path as _LCPath

_LC_RUN_ID = "948tta2-lsm-consensus-v1"
_LC_STARTED = _lc_time.monotonic()
_LC_FINISHED = False
_LC_TERMINAL = _LCPath("/kaggle/working/launcher_terminal.json")

def _lc_write_terminal(status, error=None):
    submission = _LCPath("/kaggle/working/submission.csv")
    evidence = _LCPath("/kaggle/working/candidate_evidence.json")
    payload = {
        "schema_version": 1,
        "run_id": _LC_RUN_ID,
        "status": status,
        "elapsed_seconds": round(_lc_time.monotonic() - _LC_STARTED, 3),
        "declared_budget_seconds": 43200,
        "hard_stop_seconds": 42000,
        "public_predictions_copied": False,
        "metric_hack_used": False,
        "competition_submission_performed": False,
        "submission_exists": submission.is_file(),
        "evidence_exists": evidence.is_file(),
    }
    if error is not None:
        payload["error"] = str(error)
    for name, path in (("submission", submission), ("evidence", evidence)):
        if path.is_file():
            payload[f"{name}_sha256"] = _lc_hashlib.sha256(path.read_bytes()).hexdigest()
    temporary = _LC_TERMINAL.with_suffix(".tmp")
    temporary.write_text(_lc_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(_LC_TERMINAL)

def _lc_abort_terminal():
    if not _LC_FINISHED:
        _lc_write_terminal("aborted")

def _lc_budget_expired():
    _lc_write_terminal("budget_expired")
    _lc_os._exit(124)

_lc_atexit.register(_lc_abort_terminal)
_LC_TIMER = _lc_threading.Timer(42000, _lc_budget_expired)
_LC_TIMER.daemon = True
_LC_TIMER.start()
print("Project watchdog armed for 42,000 seconds.")
'''


MODEL_SETUP = r'''# Load two independent, hash-bound LSM-FM coordinate voters.
import concurrent.futures as _lc_futures
import copy as _lc_copy
import importlib as _lc_importlib
import sys as _lc_sys

_LC_RUNTIME_MANIFEST_SHA256 = "7d3ab65f733fbe82a8a5207932cc74fcdf4581cf2bca13482c32c10572156b15"
_LC_FEATURE24_BASE_SHA256 = "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0"
_LC_FEATURE36_BASE_SHA256 = "aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7"
_LC_FEATURE24_MODEL_SHA256 = "1a87e6ed6322e0cac98d9c92f7aade2a07bc5dbd4b322c5947f17227f6e256d1"
_LC_FEATURE24_RESULT_SHA256 = "da370d810e26253b2a0bed1e16d4bdbc0264ec8a705d7e757a2607fa93c14d0d"
_LC_FEATURE24_LAUNCHER_SHA256 = "d1a56bae37975e4221256a79ecbefa7aad2cb1f7f0df7e4e87798f777372ecc6"
_LC_FEATURE36_MODEL_SHA256 = "adedfaf056ee6e53922df24dd01c5f4cf27d20f313771a37275f335c48093421"
_LC_FEATURE36_RESULT_SHA256 = "d2e2c69713edb9402a16f393cc32098bfe2cf3a9b09547cd21e3845083a3066a"
_LC_FEATURE36_LAUNCHER_SHA256 = "d68d2a816c7aa610a96ad2dfdf40546a0bf30fadeeadfd65e394586b7cf7f8ac"

def _lc_sha256(path):
    digest = _lc_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _lc_one_file_by_hash(pattern, expected_sha256, description):
    matches = sorted(
        path for path in Path("/kaggle/input").rglob(pattern)
        if path.is_file() and _lc_sha256(path) == expected_sha256
    )
    if len(matches) != 1:
        raise FileNotFoundError({description: [str(path) for path in matches]})
    return matches[0]

_LC_RUNTIME_MANIFEST_PATH = _lc_one_file_by_hash(
    "SOURCE_MANIFEST.json", _LC_RUNTIME_MANIFEST_SHA256, "LSM runtime manifest"
)
_LC_RUNTIME_ROOT = _LC_RUNTIME_MANIFEST_PATH.parent
if _lc_sha256(_LC_RUNTIME_MANIFEST_PATH) != _LC_RUNTIME_MANIFEST_SHA256:
    raise RuntimeError("LSM runtime manifest hash mismatch")
_LC_RUNTIME_MANIFEST = _lc_json.loads(
    _LC_RUNTIME_MANIFEST_PATH.read_text(encoding="utf-8")
)
for _name, _record in _LC_RUNTIME_MANIFEST["files"].items():
    _path = _LC_RUNTIME_ROOT / _name
    if _name == "monai-1.5.1.zip" and not _path.is_file():
        continue
    if _lc_sha256(_path) != _record["sha256"]:
        raise RuntimeError(f"LSM runtime file changed: {_name}")

_lc_monai_archive = _LC_RUNTIME_ROOT / "monai-1.5.1.zip"
_lc_monai_mount = _LC_RUNTIME_ROOT / _LC_RUNTIME_MANIFEST["archives"]["monai-1.5.1.zip"]["mount_directory"]
if _lc_monai_archive.is_file():
    _lc_monai_root = _lc_monai_archive
elif _lc_monai_mount.is_dir():
    _lc_monai_root = _lc_monai_mount
else:
    raise FileNotFoundError("Pinned MONAI runtime is missing")
_lc_sys.path.insert(0, str(_lc_monai_root))
_lc_sys.path.insert(0, str(_LC_RUNTIME_ROOT))

def _lc_locate_source(root_name, run_id, launcher_sha, model_sha, result_sha):
    roots = sorted(path for path in Path("/kaggle/input").rglob(root_name) if path.is_dir())
    candidates = roots if roots else [Path("/kaggle/input")]
    launchers, models, results = [], [], []
    for root in candidates:
        launchers.extend(path for path in root.rglob("launcher_terminal.json") if _lc_sha256(path) == launcher_sha)
        models.extend(path for path in root.rglob("best.pt") if _lc_sha256(path) == model_sha)
        results.extend(path for path in root.rglob("pu_training_result.json") if _lc_sha256(path) == result_sha)
    launchers, models, results = map(lambda rows: sorted(set(rows)), (launchers, models, results))
    if not (len(launchers) == len(models) == len(results) == 1):
        raise FileNotFoundError({
            "run_id": run_id,
            "launchers": [str(path) for path in launchers],
            "models": [str(path) for path in models],
            "results": [str(path) for path in results],
        })
    launcher = _lc_json.loads(launchers[0].read_text(encoding="utf-8"))
    result = _lc_json.loads(results[0].read_text(encoding="utf-8"))
    if not (
        launcher.get("run_id") == run_id
        and launcher.get("status") == "completed"
        and launcher.get("competition_submission_performed") is False
        and launcher.get("public_predictions_copied") is False
        and launcher.get("training_result_sha256") == result_sha
        and result.get("status") == "completed"
        and result.get("validation_overlap") == []
        and result.get("public_predictions_copied") is False
        and result.get("best_weight_sha256") == model_sha
    ):
        raise RuntimeError(f"Ineligible LSM source evidence: {run_id}")
    return models[0]

_LC_FEATURE24_MODEL_PATH = _lc_locate_source(
    "biohub-lsm-fm-pu-adaptation-v2", "lsm-fm-pu-adaptation-v2",
    _LC_FEATURE24_LAUNCHER_SHA256, _LC_FEATURE24_MODEL_SHA256,
    _LC_FEATURE24_RESULT_SHA256,
)
_LC_FEATURE36_MODEL_PATH = _lc_locate_source(
    "biohub-lsm-fm-image-text-pu-adaptation-v1", "lsm-fm-image-text-pu-adaptation-v1",
    _LC_FEATURE36_LAUNCHER_SHA256, _LC_FEATURE36_MODEL_SHA256,
    _LC_FEATURE36_RESULT_SHA256,
)

import torch as _lc_torch
from inference import predict_probability_batch as _lc_predict_probability_batch
from localization_refinement import refine_peaks_weighted as _lc_refine_peaks_weighted
from lsm_fm_image_only_model import build_lsm_fm_detector as _lc_build_feature24
from lsm_fm_image_text_model import build_lsm_fm_detector as _lc_build_feature36
from train_spatialdino_pu_detector import normalize_spatialdino_frame as _lc_normalize_frame

_lc_gpu_names = [_lc_torch.cuda.get_device_name(i) for i in range(_lc_torch.cuda.device_count())]
if len(_lc_gpu_names) != 2 or any("T4" not in name for name in _lc_gpu_names):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {_lc_gpu_names}")

_LC_DEVICE24 = _lc_torch.device("cuda:0")
_LC_DEVICE36 = _lc_torch.device("cuda:1")
_LC_MODEL24 = _lc_build_feature24(
    _LC_RUNTIME_ROOT / "lsm_fm_image_only_student.pt",
    expected_sha256=_LC_FEATURE24_BASE_SHA256,
)
_LC_MODEL36 = _lc_build_feature36(
    _LC_RUNTIME_ROOT / "lsm_fm_image_text_student.pt",
    expected_sha256=_LC_FEATURE36_BASE_SHA256,
)
_lc_state24 = _lc_torch.load(_LC_FEATURE24_MODEL_PATH, map_location="cpu", weights_only=True)
_lc_state36 = _lc_torch.load(_LC_FEATURE36_MODEL_PATH, map_location="cpu", weights_only=True)
_LC_MODEL24.load_state_dict(_lc_state24["state_dict"], strict=True)
_LC_MODEL36.load_state_dict(_lc_state36["state_dict"], strict=True)
if sum(p.numel() for p in _LC_MODEL24.parameters()) != 15_702_979:
    raise RuntimeError("Feature-24 LSM parameter inventory changed")
if sum(p.numel() for p in _LC_MODEL36.parameters()) != 35_072_515:
    raise RuntimeError("Feature-36 LSM parameter inventory changed")
_LC_MODEL24.requires_grad_(False).eval().to(_LC_DEVICE24)
_LC_MODEL36.requires_grad_(False).eval().to(_LC_DEVICE36)
del _lc_state24, _lc_state36
print({
    "lsm_coordinate_consensus": True,
    "feature24_model_sha256": _LC_FEATURE24_MODEL_SHA256,
    "feature36_model_sha256": _LC_FEATURE36_MODEL_SHA256,
    "gpu_names": _lc_gpu_names,
    "selection_rule": "exact_equal_final_integer_coordinate",
})
'''


COORDINATE_HELPERS = r'''
def _lc_probability_for_model(model, device, normalized_frame):
    image = _lc_torch.from_numpy(normalized_frame[None, None]).to(device)
    probability = _lc_predict_probability_batch(model, image, yx_tta=True)[0, 0]
    result = probability.cpu().numpy()
    del image, probability
    return result

def _apply_lsm_coordinate_consensus(nodes_by_id, dataset):
    stats = {
        "lsm_consensus_nodes_seen": len(nodes_by_id),
        "lsm_consensus_nodes_agreed": 0,
        "lsm_consensus_nodes_moved": 0,
        "lsm_consensus_nodes_disagreed": 0,
        "lsm_consensus_node_count_changes": 0,
        "lsm_consensus_time_or_id_changes": 0,
        "lsm_consensus_edge_changes": 0,
        "lsm_consensus_out_of_bounds": 0,
    }
    if os.environ.get("BIOHUB_LSM_CONSENSUS_ENABLE", "1") == "0":
        return nodes_by_id, stats
    if dataset is None:
        raise RuntimeError("LSM coordinate consensus requires a dataset")
    before_identity = sorted((int(node_id), int(node["t"])) for node_id, node in nodes_by_id.items())
    before_count = len(nodes_by_id)
    nodes_by_time = {}
    for node_id, node in nodes_by_id.items():
        nodes_by_time.setdefault(int(node["t"]), []).append(int(node_id))
    scale = np.asarray((1.0, 4.0, 4.0), dtype=np.float32)
    frame_cache = {}
    for timepoint, node_ids in sorted(nodes_by_time.items()):
        node_ids = sorted(node_ids)
        frame = read_test_frame(dataset, timepoint, frame_cache)
        normalized = _lc_normalize_frame(frame[:, ::4, ::4].astype(np.float32))
        with _lc_futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(_lc_probability_for_model, _LC_MODEL24, _LC_DEVICE24, normalized),
                executor.submit(_lc_probability_for_model, _LC_MODEL36, _LC_DEVICE36, normalized),
            )
            probability24, probability36 = (future.result() for future in futures)
        base_native = np.asarray(
            [[nodes_by_id[node_id][axis] for axis in ("z", "y", "x")] for node_id in node_ids],
            dtype=np.float32,
        )
        base_small = base_native / scale
        target24 = _lc_refine_peaks_weighted(
            probability24, base_small, radius=2, probability_power=2.0
        ) * scale
        target36 = _lc_refine_peaks_weighted(
            probability36, base_small, radius=2, probability_power=2.0
        ) * scale
        proposal24 = np.rint(base_native + np.float32(0.25) * (target24 - base_native)).astype(np.int64)
        proposal36 = np.rint(base_native + np.float32(0.25) * (target36 - base_native)).astype(np.int64)
        maximum = np.asarray(frame.shape, dtype=np.int64) - 1
        proposal24 = np.clip(proposal24, 0, maximum)
        proposal36 = np.clip(proposal36, 0, maximum)
        base_rounded = np.rint(base_native).astype(np.int64)
        agreement = np.all(proposal24 == proposal36, axis=1)
        moved = agreement & np.any(proposal24 != base_rounded, axis=1)
        stats["lsm_consensus_nodes_agreed"] += int(agreement.sum())
        stats["lsm_consensus_nodes_disagreed"] += int((~agreement).sum())
        stats["lsm_consensus_nodes_moved"] += int(moved.sum())
        for local_index in np.flatnonzero(moved):
            node = nodes_by_id[node_ids[int(local_index)]]
            coordinate = proposal24[int(local_index)]
            node["z"], node["y"], node["x"] = map(float, coordinate)
        del probability24, probability36, target24, target36
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint:
                del frame_cache[cached_timepoint]
    after_identity = sorted((int(node_id), int(node["t"])) for node_id, node in nodes_by_id.items())
    stats["lsm_consensus_node_count_changes"] = len(nodes_by_id) - before_count
    stats["lsm_consensus_time_or_id_changes"] = int(after_identity != before_identity)
    for node in nodes_by_id.values():
        point = np.asarray([node["z"], node["y"], node["x"]], dtype=np.float64)
        if not np.isfinite(point).all() or (point < 0).any():
            stats["lsm_consensus_out_of_bounds"] += 1
    if any(stats[key] for key in (
        "lsm_consensus_node_count_changes",
        "lsm_consensus_time_or_id_changes",
        "lsm_consensus_edge_changes",
        "lsm_consensus_out_of_bounds",
    )):
        raise RuntimeError(f"LSM coordinate-consensus integrity failure: {stats}")
    return nodes_by_id, stats

'''


VALIDATOR_LOOP_OLD = r'''    rows_this_config = []
    for stem in val_stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        pred_path = val_pred_paths.get(stem)
        if not gt_path.exists() or pred_path is None:
            print(f"VALIDATOR: skipping {stem} (missing GT or prediction .geff)")
            continue
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)

        pred_graph = graph_from_geff(pred_path)
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id, "t": int(row["t"]),
                "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        # REVIEW FIX: DeepCenter's gap/division veto path reads raw zarr
        # volume data via read_test_frame(dataset, t, ...), which resolves
        # TEST_DIR as a bare global at call time -- correct for the real
        # test-set run, but held-out validator samples live in TRAIN_DIR.
        # Redirect the global for exactly the duration of this call and
        # restore it unconditionally, even if filter_output_graph raises.
        _real_test_dir = TEST_DIR
        globals()["TEST_DIR"] = TRAIN_DIR
        try:
            processed_nodes, processed_edges, _stage_stats = filter_output_graph(
                raw_nodes_by_id, raw_edges, dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
        finally:
            globals()["TEST_DIR"] = _real_test_dir
        pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
        pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]

        row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
        row["stem"] = stem
        row["t_true_source"] = "estimated_number_of_nodes" if t_true is not None else "MISSING"
        rows_this_config.append(row)
        validator_sample_rows.append(row)

    if rows_this_config:
        summary = aggregate_official(rows_this_config)
        summary["n_samples"] = len(rows_this_config)
        validator_summary_rows.append(summary)
'''


VALIDATOR_LOOP_NEW = r'''    validator_rows_by_arm = {"control": [], "lsm_consensus": []}
    for stem in val_stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        pred_path = val_pred_paths.get(stem)
        if not gt_path.exists() or pred_path is None:
            print(f"VALIDATOR: skipping {stem} (missing GT or prediction .geff)")
            continue
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)

        pred_graph = graph_from_geff(pred_path)
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id, "t": int(row["t"]),
                "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        _real_test_dir = TEST_DIR
        _real_lsm_flag = os.environ.get("BIOHUB_LSM_CONSENSUS_ENABLE")
        globals()["TEST_DIR"] = TRAIN_DIR
        try:
            os.environ["BIOHUB_LSM_CONSENSUS_ENABLE"] = "0"
            control_nodes, control_edges, _control_stats = filter_output_graph(
                raw_nodes_by_id, raw_edges, dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            candidate_nodes = _lc_copy.deepcopy(control_nodes)
            os.environ["BIOHUB_LSM_CONSENSUS_ENABLE"] = "1"
            candidate_nodes, _coordinate_stats = _apply_lsm_coordinate_consensus(
                candidate_nodes, stem
            )
        finally:
            globals()["TEST_DIR"] = _real_test_dir
            if _real_lsm_flag is None:
                os.environ.pop("BIOHUB_LSM_CONSENSUS_ENABLE", None)
            else:
                os.environ["BIOHUB_LSM_CONSENSUS_ENABLE"] = _real_lsm_flag

        plain_edges = [(int(e["source_id"]), int(e["target_id"])) for e in control_edges]
        for arm, arm_nodes in (("control", control_nodes), ("lsm_consensus", candidate_nodes)):
            row = score_sample(
                nodes_by_id_to_plain(arm_nodes), plain_edges,
                gt_nodes_plain, gt_edges_plain, t_true,
            )
            row["stem"] = stem
            row["arm"] = arm
            row["lsm_consensus_nodes_moved"] = (
                int(_coordinate_stats["lsm_consensus_nodes_moved"])
                if arm == "lsm_consensus" else 0
            )
            row["t_true_source"] = "estimated_number_of_nodes" if t_true is not None else "MISSING"
            validator_rows_by_arm[arm].append(row)
            validator_sample_rows.append(row)

    for arm, rows_this_config in validator_rows_by_arm.items():
        if rows_this_config:
            summary = aggregate_official(rows_this_config)
            summary["arm"] = arm
            summary["n_samples"] = len(rows_this_config)
            validator_summary_rows.append(summary)
'''


EVIDENCE = r'''# Hash-bound, fail-closed comparison against the untouched control.
_lc_validator = pd.read_csv(VALIDATOR_STATS_PATH)
_lc_by_arm = {
    arm: _lc_validator[_lc_validator["arm"] == arm].copy()
    for arm in ("control", "lsm_consensus")
}
if any(len(rows) != len(val_stems) for rows in _lc_by_arm.values()):
    raise RuntimeError("Candidate/control validator coverage is incomplete")

def _lc_aggregate_frame(rows):
    weight = float(rows["weight"].sum())
    if weight <= 0:
        raise RuntimeError("Validator arm emitted zero edge weight")
    adjusted = float((rows["adjusted_edge_jaccard"] * rows["weight"]).sum() / weight)
    div_tp = int(rows["div_tp"].sum())
    div_fp = int(rows["div_fp"].sum())
    div_fn = int(rows["div_fn"].sum())
    denominator = div_tp + div_fp + div_fn
    division = div_tp / denominator if denominator else 0.0
    return {
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "proxy_score": adjusted + VALIDATOR_DIVISION_WEIGHT * division,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
    }

_lc_control = _lc_aggregate_frame(_lc_by_arm["control"])
_lc_candidate = _lc_aggregate_frame(_lc_by_arm["lsm_consensus"])
_lc_control_movie = _lc_by_arm["control"].set_index("stem")
_lc_candidate_movie = _lc_by_arm["lsm_consensus"].set_index("stem")
if set(_lc_control_movie.index) != set(_lc_candidate_movie.index):
    raise RuntimeError("Candidate/control movie sets differ")
_lc_per_movie_delta = {
    stem: float(
        _lc_candidate_movie.loc[stem, "adjusted_edge_jaccard"]
        - _lc_control_movie.loc[stem, "adjusted_edge_jaccard"]
    )
    for stem in sorted(_lc_control_movie.index)
}
_lc_production_stats = pd.read_csv(RUN_STATS_PATH)
_lc_production_moves = int(_lc_production_stats["lsm_consensus_nodes_moved"].sum())
_lc_integrity_columns = (
    "lsm_consensus_node_count_changes",
    "lsm_consensus_time_or_id_changes",
    "lsm_consensus_edge_changes",
    "lsm_consensus_out_of_bounds",
)
_lc_integrity_passed = all(int(_lc_production_stats[name].sum()) == 0 for name in _lc_integrity_columns)
_lc_eligible = bool(
    _lc_candidate["proxy_score"] > _lc_control["proxy_score"]
    and min(_lc_per_movie_delta.values()) >= 0.0
    and _lc_candidate["division_jaccard"] >= _lc_control["division_jaccard"]
    and _lc_production_moves > 0
    and _lc_integrity_passed
)
_lc_submission = Path("/kaggle/working/submission.csv")
_lc_evidence = {
    "schema_version": 1,
    "status": "eligible_for_submission" if _lc_eligible else "rejected_at_complete_movie_gate",
    "run_id": _LC_RUN_ID,
    "source_notebook": "redoctopusk/biohub-948tta2",
    "source_notebook_sha256": "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189",
    "public_lineage_attributed": True,
    "public_predictions_copied": False,
    "exact_public_replica": False,
    "metric_hack_used": False,
    "estimated_node_count_used_by_candidate": False,
    "coordinate_policy": "two_lsm_exact_final_integer_agreement_r2_p2_b025",
    "feature24_model_sha256": _LC_FEATURE24_MODEL_SHA256,
    "feature36_model_sha256": _LC_FEATURE36_MODEL_SHA256,
    "production_coordinate_changes": _lc_production_moves,
    "integrity_passed": _lc_integrity_passed,
    "control": _lc_control,
    "candidate": _lc_candidate,
    "proxy_score_delta": _lc_candidate["proxy_score"] - _lc_control["proxy_score"],
    "minimum_movie_adjusted_edge_delta": min(_lc_per_movie_delta.values()),
    "per_movie_adjusted_edge_delta": _lc_per_movie_delta,
    "complete_movie_count": len(_lc_per_movie_delta),
    "competition_submission_performed": False,
    "authorized_for_submission": _lc_eligible,
    "submission_sha256": _lc_sha256(_lc_submission),
}
Path("/kaggle/working/candidate_evidence.json").write_text(
    _lc_json.dumps(_lc_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(_lc_json.dumps(_lc_evidence, indent=2, sort_keys=True))
_LC_FINISHED = True
_LC_TIMER.cancel()
_lc_write_terminal("completed")
'''


def build_notebook() -> dict:
    if sha256_file(SOURCE_NOTEBOOK) != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("Audited 0.948-TTA2 notebook source changed")
    if sha256_file(SOURCE_METADATA) != SOURCE_METADATA_SHA256:
        raise RuntimeError("Audited 0.948-TTA2 metadata changed")
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    config_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if 'os.environ["BIOHUB_DET_THRESHOLD"]' in "".join(cell.get("source", []))
    )
    config = "".join(notebook["cells"][config_index]["source"])
    config = replace_exact(
        config,
        'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"',
        'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"\n'
        'os.environ["BIOHUB_LSM_CONSENSUS_ENABLE"] = "1"',
    )
    notebook["cells"][config_index]["source"] = config.splitlines(keepends=True)

    inference_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "Fail fast instead of silently running volumetric inference on CPU"
        in "".join(cell.get("source", []))
    )
    notebook["cells"][inference_index + 1 : inference_index + 1] = [
        markdown_cell(ATTRIBUTION),
        code_cell(MODEL_SETUP),
    ]

    post_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "def filter_output_graph(" in "".join(cell.get("source", []))
    )
    post = "".join(notebook["cells"][post_index]["source"])
    post = replace_exact(
        post,
        "def filter_output_graph(\n",
        COORDINATE_HELPERS + "def filter_output_graph(\n",
    )
    post = replace_exact(
        post,
        '    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n'
        '    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")',
        '    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n'
        '    nodes_by_id, _lc_stats = _apply_lsm_coordinate_consensus(nodes_by_id, dataset)\n'
        '    stats.update(_lc_stats)\n'
        '    print(f"  [{dataset}] LSM consensus moved {stats[\'lsm_consensus_nodes_moved\']} nodes")\n'
        '    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")',
    )
    notebook["cells"][post_index]["source"] = post.splitlines(keepends=True)

    validator_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "VALIDATOR: scoring against the official metric" in "".join(cell.get("source", []))
    )
    validator = "".join(notebook["cells"][validator_index]["source"])
    validator = replace_exact(validator, VALIDATOR_LOOP_OLD, VALIDATOR_LOOP_NEW)
    notebook["cells"][validator_index]["source"] = validator.splitlines(keepends=True)
    notebook["cells"][validator_index + 1 : validator_index + 1] = [code_cell(EVIDENCE)]

    notebook["cells"][config_index:config_index] = [code_cell(WATCHDOG)]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook()
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub 948TTA2 LSM Consensus v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "lsm-fm", "non-replica"],
        "dataset_sources": [*base_metadata["dataset_sources"], RUNTIME_REF],
        "kernel_sources": [FEATURE24_KERNEL, FEATURE36_KERNEL],
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
