from dataclasses import dataclass
from pathlib import Path

import pytest

from research.peak_rank_detection import (
    train_expanded_real_temporal_stable_hard_mined_safe_rank_detector as hard,
)


@dataclass
class Example:
    identity: str


def _manifest() -> dict:
    entries = []
    for embryo, count, target in (("44b6", 150, 495), ("6bba", 330, 495)):
        base, extra = divmod(target, count)
        for index in range(count):
            entries.append(
                {
                    "identity": f"{embryo}_{index:03d}:t1",
                    "embryo": embryo,
                    "sampling_multiplicity": base + int(index < extra),
                }
            )
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": "optimization-only-balanced-local-snr-hard-mining-v27",
        "expected_unique_counts": {"44b6": 150, "6bba": 330},
        "effective_counts": {"44b6": 495, "6bba": 495},
        "target_effective_count_per_embryo": 495,
        "local_snr_bands": [[3, 9], [3, 11], [5, 13]],
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "selection_data_read": False,
        "sealed_audit_data_read": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "entries": entries,
    }


def test_manifest_repetition_is_exact_and_balanced() -> None:
    manifest = _manifest()
    examples = [Example(row["identity"]) for row in manifest["entries"]]
    repeated, receipt = hard.repeat_examples_by_manifest(examples, manifest)
    assert len(repeated) == 990
    assert receipt["effective_counts"] == {"44b6": 495, "6bba": 495}
    assert receipt["multiplicity_histogram"] == {
        "44b6": {"3": 105, "4": 45},
        "6bba": {"1": 165, "2": 165},
    }


def test_manifest_repetition_rejects_heldout_or_count_drift() -> None:
    manifest = _manifest()
    examples = [Example(row["identity"]) for row in manifest["entries"]]
    examples[0] = Example("44b6_unexpected:t1")
    with pytest.raises(ValueError, match="differ"):
        hard.repeat_examples_by_manifest(examples, manifest)
    manifest = _manifest()
    manifest["selection_data_read"] = True
    with pytest.raises(ValueError, match="clean contract"):
        hard.repeat_examples_by_manifest(
            [Example(row["identity"]) for row in manifest["entries"]], manifest
        )


def test_trainer_declares_optimization_only_sampling_contract() -> None:
    source = Path(
        "research/peak_rank_detection/"
        "train_expanded_real_temporal_stable_hard_mined_safe_rank_detector.py"
    ).read_text(encoding="utf-8")
    assert "expanded_real_optimization_only" in source
    assert '"hard_mining_selection_sampling_changed": False' in source
    assert '"hard_mining_sealed_audit_sampling_changed": False' in source
    assert hard.SAMPLING_MANIFEST_SHA256 == (
        "9967efa25021453b4f043a23e47e744153da678f17f5c00180ecd6ebd4dca900"
    )
