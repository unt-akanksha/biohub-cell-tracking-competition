from research.peak_rank_detection import (
    train_expanded_real_balanced_safe_rank_detector as balanced,
)
from research.peak_rank_detection import (
    train_expanded_real_xl_balanced_safe_rank_detector as xl,
)


def test_xl_wrapper_changes_only_run_identity(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(xl.balanced, "main", lambda: called.append(True))
    original = balanced.RUN_ID
    try:
        xl.main()
        assert called == [True]
        assert balanced.RUN_ID == xl.RUN_ID
        assert xl.RUN_ID.endswith("safe-rank-peak-rank-v21")
        assert balanced.BALANCED_OPTIMIZATION_COUNTS == {"44b6": 300, "6bba": 330}
    finally:
        balanced.RUN_ID = original
