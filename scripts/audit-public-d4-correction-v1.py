"""Build a source-only public-base correction and its label-free audit receipt."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from public_d4_correction import (  # noqa: E402
    NOTEBOOK_SHA256, cell_text, correct_antidiagonal, corrected_notebook,
    geometry_check, materialize_public_predictor, sha256,
)


def main() -> None:
    started = time.monotonic()
    path = ROOT / ".biohub/cache/public-frontier-20260910-2132/lf-dctta/biohub-lf-dctta.ipynb"
    support = ROOT / ".biohub/cache/datasets/biohub-support-source/predict_unet_transformer.py"
    destination = ROOT / ".biohub/cache/public-d4-correction-v1"
    report = ROOT / "reports/experiments/public-d4-correction-v1-audit.json"
    if destination.exists() or report.exists():
        raise ValueError("Refusing to overwrite a frozen source-audit artifact")
    source_bytes = path.read_bytes()
    if sha256(source_bytes) != NOTEBOOK_SHA256:
        raise ValueError("Pinned public notebook SHA256 mismatch")
    notebook = json.loads(source_bytes)
    legacy = materialize_public_predictor(notebook, support.read_bytes())
    corrected, predictor_edits = correct_antidiagonal(legacy, predictor=True)
    dc, dc_edits = correct_antidiagonal(cell_text(notebook, 5), predictor=False)
    proof = geometry_check()
    candidate = corrected_notebook(notebook, legacy, corrected, dc)
    files = {
        "public-predictor-original.py": legacy,
        "public-predictor-d4-corrected.py": corrected,
        "public-postprocess-d4-corrected.py": dc,
        "biohub-public-d4-research-only.ipynb": json.dumps(candidate, indent=1, ensure_ascii=False) + "\n",
    }
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, value in files.items():
        data = value.encode("utf-8")
        (destination / name).write_bytes(data)
        manifest[name] = dict(sha256=sha256(data), bytes=len(data))
    result = dict(
        run_id="public-d4-correction-v1", status="source_geometry_audit_passed",
        source_url="https://www.kaggle.com/code/sjlee101/biohub-lf-dctta",
        source_version=1, source_notebook_sha256=NOTEBOOK_SHA256,
        source_code_sha256=sha256((ROOT / "research/public_d4_correction.py").read_bytes()),
        runner_sha256=sha256(Path(__file__).read_bytes()),
        support_sha256=sha256(support.read_bytes()),
        artifact_root=destination.relative_to(ROOT).as_posix(), artifacts=manifest,
        predictor_edits=predictor_edits, deepcenter_edits=dc_edits, geometry=proof,
        notebook_executed=False, public_predictions_downloaded=False,
        ground_truth_opened=False, new_target_movies_opened=0,
        public_current_best_score_verified=False,
        runtime_license_audit_complete=False, real_model_preflight_passed=False,
        official_complete_movie_scoring_passed=False, independently_held_out=False,
        authorized_for_submission=False, gpu_seconds=0,
        elapsed_seconds=time.monotonic() - started,
    )
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
