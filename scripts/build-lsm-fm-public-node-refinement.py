from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "lsm-fm-public-node-refinement-v1"
KAGGLE_SLUG = "biohub-lsm-fm-public-node-refine-v1"
RUNTIME_NAME = "biohub-lsm-fm-public-node-runtime-v1"
RUNTIME = ROOT / ".biohub" / "staging" / RUNTIME_NAME
TEMPLATE = (
    ROOT
    / "kaggle"
    / "biohub-lsm-fm-boundary-refinement-v1"
    / "biohub-lsm-fm-boundary-refinement-v1.ipynb"
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
        raise FileNotFoundError("build the public-node runtime before the kernel")
    manifest_sha256 = sha256_file(manifest_path)
    notebook = json.loads(TEMPLATE.read_text(encoding="ascii"))
    if len(notebook["cells"]) != 5:
        raise ValueError("boundary-refinement template cell layout changed")

    watchdog = source(notebook["cells"][0])
    replacements = {
        'RUN_ID = "lsm-fm-boundary-refinement-v1"': f'RUN_ID = "{RUN_ID}"',
        'Path("/kaggle/working/lsm_fm_boundary/lsm_fm_boundary_refinement.json")': (
            'Path("/kaggle/working/lsm_fm_public_node/lsm_fm_public_node_refinement.json")'
        ),
        'print("LSM-FM boundary-refinement watchdog armed for 3,300 seconds.")': (
            'print("LSM-FM public-node-refinement watchdog armed for 3,300 seconds.")'
        ),
    }
    for old, new in replacements.items():
        if watchdog.count(old) != 1:
            raise ValueError(f"watchdog template changed: {old}")
        watchdog = watchdog.replace(old, new)
    provenance = '        "public_predictions_copied": False,\n'
    replacement = (
        '        "public_graph_used_as_base": True,\n'
        '        "exact_public_replica": False,\n'
    )
    if watchdog.count(provenance) != 1:
        raise ValueError("watchdog provenance template changed")
    watchdog = watchdog.replace(provenance, replacement)
    set_source(notebook["cells"][0], watchdog)

    set_source(
        notebook["cells"][1],
        """
        # Independent LSM-FM refinement of frozen public graph nodes

        This private validation run preserves every public node identifier,
        node count, and edge while testing global coordinate blends toward the
        independently trained feature-36 LSM-FM probability field. Selection
        uses eight disjoint movies and requires a strict matched-node gain.
        Four acceptance movies stay sealed unless that gate passes. There is no
        leaderboard feedback and no submission path.
        """,
    )

    setup = source(notebook["cells"][2])
    old_hash_line = next(
        line for line in setup.splitlines() if line.startswith("RUNTIME_MANIFEST_SHA256 = ")
    )
    replacements = {
        old_hash_line: f'RUNTIME_MANIFEST_SHA256 = "{manifest_sha256}"',
        "biohub-lsm-fm-boundary-runtime-v1": RUNTIME_NAME,
        "boundary runtime manifest hash mismatch": "public-node runtime manifest hash mismatch",
    }
    for old, new in replacements.items():
        if old not in setup:
            raise ValueError(f"setup template changed: {old}")
        setup = setup.replace(old, new)
    set_source(notebook["cells"][2], setup)

    set_source(
        notebook["cells"][3],
        r'''
        output_dir = Path("/kaggle/working/lsm_fm_public_node")
        command = [
            sys.executable,
            str(runtime / "evaluate_public_node_refinement.py"),
            "--competition-dir", str(competition),
            "--lsm-fm-checkpoint", str(feature36_base),
            "--lsm-fm-checkpoint-sha256", FEATURE36_BASE_SHA256,
            "--model-path", str(feature36_model),
            "--training-result", str(feature36_result),
            "--public-predictions", str(graph_runtime / "validator_raw"),
            "--output-dir", str(output_dir),
            "--batch-size", "1",
            "--max-wall-seconds", "3000",
        ]
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = os.pathsep.join([
            str(runtime), str(monai_import_root), str(support_repo.parent),
            run_env.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep)
        print("Launching clean LSM-FM public-node refinement:", " ".join(command))
        try:
            subprocess.run(command, check=True, env=run_env)
        except Exception as exc:
            write_terminal("validation_failed", exc)
            raise
        result_path = output_dir / "lsm_fm_public_node_refinement.json"
        if not result_path.is_file():
            raise RuntimeError("public-node evaluator exited without terminal evidence")
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
    old = 'print("LSM-FM boundary refinement complete; no submission was created.")'
    new = 'print("LSM-FM public-node refinement complete; no submission was created.")'
    if finish.count(old) != 1:
        raise ValueError("finish template changed")
    set_source(notebook["cells"][4], finish.replace(old, new))

    TARGET.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": f"indarkarhana/{KAGGLE_SLUG}",
        "title": "Biohub LSM-FM Public Node Refinement v1",
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
