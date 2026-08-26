from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "lsm-fm-boundary-refinement-v1"
KAGGLE_SLUG = "biohub-lsm-fm-boundary-refine-v1"
RUNTIME_NAME = "biohub-lsm-fm-boundary-runtime-v1"
RUNTIME = ROOT / ".biohub" / "staging" / RUNTIME_NAME
TEMPLATE = (
    ROOT
    / "kaggle"
    / "biohub-lsm-fm-image-text-localization-refinement-v1"
    / "biohub-lsm-fm-image-text-localization-refinement-v1.ipynb"
)
TARGET = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = TARGET / f"biohub-{RUN_ID}.ipynb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, value: str) -> None:
    cell["source"] = textwrap.dedent(value).lstrip().splitlines(keepends=True)


def main() -> None:
    manifest_path = RUNTIME / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("build the boundary runtime before the kernel")
    manifest_sha256 = sha256_file(manifest_path)
    notebook = json.loads(TEMPLATE.read_text(encoding="ascii"))
    if len(notebook["cells"]) != 5:
        raise ValueError("localization-refinement template cell layout changed")

    watchdog = source(notebook["cells"][0])
    replacements = {
        'RUN_ID = "lsm-fm-image-text-localization-refinement-v1"': f'RUN_ID = "{RUN_ID}"',
        'Path("/kaggle/working/lsm_fm_refinement/lsm_fm_localization_refinement.json")': (
            'Path("/kaggle/working/lsm_fm_boundary/lsm_fm_boundary_refinement.json")'
        ),
        'print("LSM-FM localization-refinement watchdog armed for 3,300 seconds.")': (
            'print("LSM-FM boundary-refinement watchdog armed for 3,300 seconds.")'
        ),
    }
    for old, new in replacements.items():
        if watchdog.count(old) != 1:
            raise ValueError(f"watchdog template changed: {old}")
        watchdog = watchdog.replace(old, new)
    set_source(notebook["cells"][0], watchdog)

    set_source(
        notebook["cells"][1],
        """
        # Near-gate feature-36 centroid boundary sweep

        The previous clean run found a global radius-2 probability centroid
        that improved all eight selection movies and missed the worst-movie
        gate by one match. This private inference-only fallback tests six
        predeclared nearby probability-centroid strategies. Peak identities,
        confidences, and density calibration remain fixed. Four acceptance
        movies stay sealed unless the unchanged selection gate passes. There
        is no leaderboard feedback and no submission path.
        """,
    )

    setup = source(notebook["cells"][2])
    old_runtime_hash = (
        'RUNTIME_MANIFEST_SHA256 = '
        '"81c70961de251cbd20442cd618fba57b972df7b52de113a2ad9822357b0f3fc1"'
    )
    replacements = {
        old_runtime_hash: f'RUNTIME_MANIFEST_SHA256 = "{manifest_sha256}"',
        "biohub-lsm-fm-ensemble-runtime-v1": RUNTIME_NAME,
        "ensemble runtime manifest hash mismatch": "boundary runtime manifest hash mismatch",
    }
    for old, new in replacements.items():
        if old not in setup:
            raise ValueError(f"setup template changed: {old}")
        setup = setup.replace(old, new)
    set_source(notebook["cells"][2], setup)

    set_source(
        notebook["cells"][3],
        r'''
        output_dir = Path("/kaggle/working/lsm_fm_boundary")
        command = [
            sys.executable,
            str(runtime / "evaluate_boundary_refinement.py"),
            "--competition-dir", str(competition),
            "--lsm-fm-checkpoint", str(feature36_base),
            "--lsm-fm-checkpoint-sha256", FEATURE36_BASE_SHA256,
            "--model-path", str(feature36_model),
            "--training-result", str(feature36_result),
            "--baseline-predictions", str(graph_runtime / "validator_raw"),
            "--output-dir", str(output_dir),
            "--batch-size", "1",
            "--calibration-frames", "12",
            "--max-wall-seconds", "3000",
        ]
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = os.pathsep.join([
            str(runtime), str(monai_import_root), str(support_repo.parent),
            run_env.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep)
        print("Launching clean LSM-FM boundary refinement:", " ".join(command))
        try:
            subprocess.run(command, check=True, env=run_env)
        except Exception as exc:
            write_terminal("validation_failed", exc)
            raise
        result_path = output_dir / "lsm_fm_boundary_refinement.json"
        if not result_path.is_file():
            raise RuntimeError("boundary evaluator exited without terminal evidence")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps({
            "selected_strategy": result["selected_strategy"],
            "selection_passed": result["selection_passed"],
            "acceptance_opened": result["acceptance_opened"],
            "promotion_passed": result["promotion_passed"],
        }, indent=2, sort_keys=True))
        ''',
    )

    finish = source(notebook["cells"][4])
    old = 'print("LSM-FM localization refinement complete; no submission was created.")'
    new = 'print("LSM-FM boundary refinement complete; no submission was created.")'
    if finish.count(old) != 1:
        raise ValueError("finish template changed")
    set_source(notebook["cells"][4], finish.replace(old, new))

    TARGET.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": f"indarkarhana/{KAGGLE_SLUG}",
        "title": "Biohub LSM-FM Boundary Refinement v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "swinunetr", "localization", "clean-validation"],
        "dataset_sources": [
            f"indarkarhana/{RUNTIME_NAME}",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-lsm-fm-pu-adaptation-v2",
            "indarkarhana/biohub-lsm-fm-image-text-pu-adaptation-v1",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
