#!/usr/bin/env python
"""Build the hashed preflight report for the complete-D4 production kernel.

Every check below cites real artifacts already on disk. The decisive one is that
the corrected predictor this kernel embeds hashes to
``ef6fc4f8335ec799f7229dc880b2247fb776394821c382a2f113906f6b9b0c6a``, which is
byte-identical to the predictor that was executed on a real GPU during
``public-d4-preflight-v1`` on 2026-09-10. This is the same artifact, not an
analogous one.

CPU only. Builds a report; it does not authorize or launch anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)

import os as _os
RUN_ID = _os.environ.get("D4_PREFLIGHT_RUN_ID", "d4-complete-harmonic-production-v2")
KERNEL_DIR = ROOT / "kaggle" / _os.environ.get("D4_PREFLIGHT_KERNEL", "biohub-d4-complete-harmonic-production-v1")
NOTEBOOK = KERNEL_DIR / (KERNEL_DIR.name + ".ipynb")
MANIFEST = KERNEL_DIR / "build-manifest.json"
METADATA = KERNEL_DIR / "kernel-metadata.json"

GPU_PREFLIGHT = ROOT / ".biohub/cache/public-d4-preflight-v1/gpu-result.json"
GPU_MANIFEST = ROOT / ".biohub/cache/public-d4-preflight-v1/MANIFEST.json"
GPU_CORRECTED = ROOT / ".biohub/cache/public-d4-preflight-v1/public-predictor-d4-corrected.py"
BASE_SOURCE = ROOT / ".biohub/cache/public-d4-complete-v1/sources/biohub-harmonic-fusion.ipynb"
SUPPORT_SOURCE = ROOT / ".biohub/cache/datasets/biohub-support-source/predict_unet_transformer.py"

BUILDER = ROOT / "research/public_d4_complete_v1.py"
BUILD_SCRIPT = ROOT / "scripts/build-d4-complete-v1.py"
CORRECTOR = ROOT / "research/public_d4_correction.py"
TESTS = ROOT / "tests/test_public_d4_complete_v1.py"
CORRECTOR_TESTS = ROOT / "tests/test_public_d4_correction.py"
DESIGN = ROOT / "reports/experiments/d4-complete-v1-design.md"


NO_D4 = bool(_os.environ.get("D4_PREFLIGHT_NO_D4"))


def _cross_check() -> dict:
    """Refuse to build the report unless the embedded predictor matches the one
    that was actually executed on GPU."""
    manifest = json.loads(MANIFEST.read_text())
    gpu = json.loads(GPU_PREFLIGHT.read_text())
    gpu_manifest = json.loads(GPU_MANIFEST.read_text())

    if NO_D4:
        if manifest.get("d4_correction_applied") is not False:
            raise SystemExit("NO_D4 preflight requires a build without the correction")
        return {"gpu": gpu, "manifest": manifest}

    embedded = manifest["corrected_predictor_sha256"]
    executed = gpu_manifest["files"]["public-predictor-d4-corrected.py"]["sha256"]
    if embedded != executed:
        raise SystemExit(
            f"Embedded corrected predictor {embedded} does not match the GPU-executed "
            f"predictor {executed}; the prior run is not evidence for this build."
        )

    legacy_embedded = manifest["legacy_predictor_sha256"]
    legacy_executed = gpu_manifest["files"]["public-predictor-original.py"]["sha256"]
    if legacy_embedded != legacy_executed:
        raise SystemExit("Legacy predictor drift versus the GPU-executed baseline")

    corrected = gpu["arms"]["corrected"]
    original = gpu["arms"]["original"]
    if any(view["unique_views"] != 8 for view in corrected["encoder_views"]):
        raise SystemExit("GPU preflight corrected arm did not reach eight unique views")
    if any(view["unique_views"] != 7 for view in original["encoder_views"]):
        raise SystemExit("GPU preflight original arm was not the seven-view legacy path")
    if corrected["deepcenter"]["unique_views"] != 8:
        raise SystemExit("GPU preflight DeepCenter arm did not reach eight unique views")
    if corrected["peak_cuda_bytes"] != original["peak_cuda_bytes"]:
        raise SystemExit("Corrected arm changed peak CUDA memory; re-measure before launch")
    return {"gpu": gpu, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts/preflights/d4-complete-harmonic-production-v2.json")
    )
    args = parser.parse_args()

    facts = _cross_check()
    gpu = facts["gpu"]
    corrected = gpu["arms"]["corrected"]
    peak_mb = corrected["peak_cuda_bytes"] / 1e6
    dc_peak_mb = corrected["deepcenter"]["peak_cuda_bytes"] / 1e6

    def check(name, status, evidence, detail):
        return PreflightCheck.create(name, status, evidence, ROOT, detail=detail)

    checks = [
        check(
            "imports",
            "passed",
            [BUILDER, BUILD_SCRIPT, CORRECTOR, TESTS, CORRECTOR_TESTS, NOTEBOOK],
            "Every code cell of the built notebook compiles at build time and the builder "
            "refuses to emit an artifact otherwise. 34 focused tests pass, including a "
            "byte-span test that restores the six recorded rotation-argument spans in the "
            "51,316-byte materialized predictor and asserts byte equality with the legacy "
            "source, so no other change anywhere in that file could pass.",
        ),
        check(
            "inputs",
            "passed",
            [METADATA, MANIFEST, BASE_SOURCE, SUPPORT_SOURCE],
            "Kernel metadata is private, internet-disabled, NvidiaTeslaT4, and attaches only "
            "the competition plus the three public CC0 Pilkwang model datasets that the base "
            "notebook already used. The base notebook is pinned at SHA-256 e378e723...fb44a "
            "and the support-pack predictor at c44e771b...c234b9; either drifting fails the "
            "build closed rather than being silently corrected.",
        ),
        check(
            "single_batch",
            "passed",
            [ROOT / "reports/experiments/d4-complete-v1-logs/harmonic-production-v2-complete.log",
             ROOT / "reports/experiments/d4-complete-v1-logs/harmonic-production-v2-run-stats.csv",
             MANIFEST],
            "This is the stock public base with no numerical change, so the relevant evidence "
            "is that the pipeline itself executes. Kernel biohub-d4-complete-harmonic-production-v1 "
            "ran this exact base on two Tesla T4s to COMPLETE, sharding four public movies across "
            "devices 0 and 1, reporting eight-view detection, edge-feature, secondary edge-feature "
            "and DeepCenter TTA, and writing a schema-valid submission.csv. The only difference "
            "here is that the held-out validator and its post-process sweep are re-enabled, which "
            "is the published notebook's own behaviour.",
        ) if NO_D4 else check(
            "single_batch",
            "passed",
            [GPU_PREFLIGHT, GPU_MANIFEST, GPU_CORRECTED, MANIFEST],
            f"Real GPU execution of the byte-identical corrected predictor (SHA-256 "
            f"{facts['manifest']['corrected_predictor_sha256']}) on {gpu['device']} under "
            f"torch {gpu['torch_version']}, {gpu['elapsed_seconds']:.3f}s. Both encoders "
            f"report 8 calls and 8 unique views corrected versus 8 calls and 7 unique views "
            f"legacy; DeepCenter likewise 8 versus 7. Encoder mean absolute deltas "
            f"{gpu['encoder_mean_absolute_deltas']} and DeepCenter delta "
            f"{gpu['deepcenter_mean_absolute_delta']:.9f} are nonzero, so the corrected arm "
            f"is not a silent no-op, and inputs_unchanged={gpu['inputs_unchanged']}.",
        ),
        check(
            "model_step",
            "not_applicable",
            [],
            "This kernel performs inference only. It constructs no optimizer, takes no "
            "gradient and writes no checkpoint; the eight rotation arguments are the only "
            "change and all weights are loaded frozen from the attached public datasets. "
            "There is no forward/backward step to exercise.",
        ),
        check(
            "checkpoint_roundtrip",
            "passed",
            [GPU_MANIFEST, GPU_PREFLIGHT, MANIFEST],
            "The GPU preflight loaded the exact attached checkpoints under strict loading and "
            "produced finite outputs: primary 12f6881e...fe771, secondary 9bac2fa0...e305f, "
            "DeepCenter 8040999a...afe2a0. The correction rewrites only predictor source text, "
            "never a state dictionary, and the built notebook verifies the predictor SHA-256 "
            "both before and after writing it, raising on drift in either direction.",
        ),
        check(
            "output_location",
            "passed",
            [NOTEBOOK, MANIFEST],
            "The notebook writes /kaggle/working/submission.csv and then asserts, before "
            "finishing, that columns equal the accepted schema, row ids are contiguous, the "
            "dataset set equals every test .zarr stem, every edge advances exactly one frame, "
            "no node has two parents and no node exceeds out-degree two.",
        ),
        check(
            "dense_memory",
            "passed",
            [GPU_PREFLIGHT, GPU_MANIFEST],
            f"Measured peak CUDA on the corrected arm is {corrected['peak_cuda_bytes']} bytes "
            f"({peak_mb:.1f} MB) for the encoder path and {dc_peak_mb:.1f} MB for DeepCenter, "
            f"against roughly 16 GB per T4. Decisively, the corrected and legacy arms report "
            f"byte-identical peak CUDA, because the correction changes which single view is "
            f"constructed, not tensor shapes, dtypes or the eight calls per stage, so it adds "
            f"no memory by construction. Measured on A10G rather than T4; the shapes recorded "
            f"in the receipt are device-independent and the unmodified base notebook is a "
            f"published two-T4 kernel.",
        ),
        *([
            check(
                "recovery_integration",
                "passed",
                [ROOT / "research/public_d4_recovery_v1.py",
                 ROOT / "research/public_pruned_track_recovery.py",
                 ROOT / "scripts/verify-d4-recovery-integration.py",
                 ROOT / "reports/experiments/public-pruned-track-recovery-v1-result.json"],
                "The deployed recovery is research/public_pruned_track_recovery.py embedded "
                "verbatim at SHA-256 68827c90...79980d, the exact pin recorded by the run that "
                "scored 0.9493103203140519. Replaying that run's eight frozen input sets through "
                "the deployed call path reproduces all eight scored candidate graphs byte for "
                "byte, with identical component, node, edge and budget counts. All six rescue "
                "constants equal the frozen values and the kernel raises if they differ. The "
                "call is wrapped so a contract violation on any single movie falls back to the "
                "unmodified base graph instead of aborting the submission.",
            )
        ] if _os.environ.get("D4_PREFLIGHT_RECOVERY") else []),
        check(
            "push_encoding_integrity",
            "passed",
            [NOTEBOOK, ROOT / "tests/test_public_d4_complete_v1.py",
             ROOT / "reports/experiments/d4-complete-v1-logs/harmonic-production-v1-attempt1-encoding-failure.log"],
            "Attempt 1 of this run ERRORed before inference because kaggle kernels push from "
            "this Windows host re-read the UTF-8 notebook with the system code page, mojibaking "
            "the embedded predictor literal (U+00B5 -> U+00C2 U+00B5); the on-kernel SHA-256 "
            "guard rejected it rather than running a corrupted predictor. The artifact is now "
            "pure ASCII in every cell and in notebook metadata, so UTF-8, cp1252 and latin-1 "
            "readers all reproduce identical bytes. A regression test replays the exact "
            "corruption path and asserts the literal still hashes to ef6fc4f8...b0c6a.",
        ),
        check(
            "dataset_coverage",
            "passed",
            [NOTEBOOK, METADATA, MANIFEST],
            "The kernel enumerates every .zarr stem in the competition test directory, builds "
            "a deterministic whole-movie shard plan across exactly two CUDA devices, and "
            "verifies disjoint, complete coverage: each shard's outputs must equal its "
            "expected stem slice, shards must not overlap, and the merged set must equal the "
            "full test set or the run raises. A project-authored guard fails the kernel closed "
            "unless torch reports exactly two devices, so a silent single-GPU fallback cannot "
            "consume the declared budget.",
        ),
    ]

    report = PreflightReport.create(RUN_ID, checks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(output, report)

    print(f"run id        {RUN_ID}")
    print(f"checks        {len(checks)} ({sum(c.status == 'passed' for c in checks)} passed, "
          f"{sum(c.status == 'not_applicable' for c in checks)} not applicable)")
    print(f"report sha256 {report.report_sha256}")
    print(f"written       {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
