from research.peak_rank_detection import (
    train_expanded_real_balanced_safe_rank_detector as balanced,
)
from research.peak_rank_detection import (
    train_expanded_real_temporal_stable_balanced_safe_rank_detector as stable,
)
from research.peak_rank_detection.model_temporal_stable import (
    MODEL_FAMILY,
    TemporalMinimumLocalSnrSafeRankDetector,
)


def test_temporal_stable_trainer_installs_owned_model_contract(monkeypatch) -> None:
    seen = {}

    # Register the globals with monkeypatch so this module-level composition
    # cannot leak into later trainer tests in the same pytest process.
    monkeypatch.setattr(balanced, "RUN_ID", balanced.RUN_ID)
    monkeypatch.setattr(balanced.safe, "MODEL_FAMILY", balanced.safe.MODEL_FAMILY)
    monkeypatch.setattr(
        balanced.safe,
        "SafeRankMultiscaleBlobGlobalDetector",
        balanced.safe.SafeRankMultiscaleBlobGlobalDetector,
    )

    def fake_main() -> None:
        seen["run_id"] = balanced.RUN_ID
        seen["model_family"] = balanced.safe.MODEL_FAMILY
        seen["model_class"] = balanced.safe.SafeRankMultiscaleBlobGlobalDetector

    monkeypatch.setattr(balanced, "main", fake_main)
    stable.main()
    assert seen == {
        "run_id": stable.RUN_ID,
        "model_family": MODEL_FAMILY,
        "model_class": TemporalMinimumLocalSnrSafeRankDetector,
    }
