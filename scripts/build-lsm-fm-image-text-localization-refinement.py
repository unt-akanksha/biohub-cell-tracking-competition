from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "lsm-fm-image-text-localization-refinement-v1"
KAGGLE_SLUG = "biohub-lsm-fm-f36-refinement-v1"
TEMPLATE = (
    ROOT
    / "kaggle"
    / "biohub-lsm-fm-ensemble-validation-v1"
    / "biohub-lsm-fm-ensemble-validation-v1.ipynb"
)
TARGET = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = TARGET / f"biohub-{RUN_ID}.ipynb"


def source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, value: str) -> None:
    cell["source"] = textwrap.dedent(value).lstrip().splitlines(keepends=True)


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="ascii"))
    if len(notebook["cells"]) != 5:
        raise ValueError("ensemble template cell layout changed")

    watchdog = source(notebook["cells"][0])
    replacements = {
        'RUN_ID = "lsm-fm-ensemble-validation-v1"': f'RUN_ID = "{RUN_ID}"',
        'Path("/kaggle/working/lsm_fm_ensemble/lsm_fm_ensemble_validation.json")': (
            'Path("/kaggle/working/lsm_fm_refinement/'
            'lsm_fm_localization_refinement.json")'
        ),
        'print("LSM-FM ensemble watchdog armed for 3,300 seconds.")': (
            'print("LSM-FM localization-refinement watchdog armed for 3,300 seconds.")'
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
        # Clean feature-36 LSM-FM localization refinement

        This private, inference-only experiment evaluates eight predeclared
        global coordinate refiners for the independently trained feature-36
        LSM-FM detector. It preserves peak identities, confidence ranking, and
        density calibration. Selection uses eight movies; four acceptance
        movies remain sealed unless the unchanged worst-movie gate passes.
        There is no leaderboard feedback and no submission path.
        """,
    )

    setup = source(notebook["cells"][2])
    setup = setup.replace(
        'print("LSM-FM ensemble clean validation complete; no submission was created.")',
        'print("LSM-FM localization refinement complete; no submission was created.")',
    )
    set_source(notebook["cells"][2], setup)

    set_source(
        notebook["cells"][3],
        r'''
        output_dir = Path("/kaggle/working/lsm_fm_refinement")
        command = [
            sys.executable,
            str(runtime / "evaluate_localization_refinement.py"),
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
        print("Launching clean LSM-FM localization refinement:", " ".join(command))
        try:
            subprocess.run(command, check=True, env=run_env)
        except Exception as exc:
            write_terminal("validation_failed", exc)
            raise
        result_path = output_dir / "lsm_fm_localization_refinement.json"
        if not result_path.is_file():
            raise RuntimeError("refinement evaluator exited without terminal evidence")
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
    old = 'print("LSM-FM ensemble clean validation complete; no submission was created.")'
    new = 'print("LSM-FM localization refinement complete; no submission was created.")'
    if finish.count(old) != 1:
        raise ValueError("finish template changed")
    set_source(notebook["cells"][4], finish.replace(old, new))

    notebook["metadata"]["kaggle"]["accelerator"] = "gpu"
    TARGET.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{KAGGLE_SLUG}",
        "title": "Biohub LSM-FM F36 Refinement v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "swinunetr", "localization", "clean-validation"],
        "dataset_sources": [
            "indarkarhana/biohub-lsm-fm-ensemble-runtime-v1",
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
