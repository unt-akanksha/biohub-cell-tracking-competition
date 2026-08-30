from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-ranked-consensus-division-dataset.py"


def test_builder_declares_only_project_runtime_and_no_submission_command() -> None:
    module = runpy.run_path(str(BUILDER))

    assert module["DATASET_ID"] == "indarkarhana/biohub-ranked-consensus-division-v1"
    assert set(module["RUNTIME_FILES"]) == {
        "learned_division_recovery.py",
        "multiscale_contextual_pair_fusion.py",
        "patch_model.py",
        "handcrafted_division.py",
    }
    for path in module["RUNTIME_FILES"].values():
        assert path.is_file()
    assert module["SKLEARN_VERSION"] == "1.9.0"
    assert "cp312-cp312" in module["SKLEARN_WHEEL_NAME"]
