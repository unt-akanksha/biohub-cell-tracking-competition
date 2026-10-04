from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from public_d4_numpy_execution import encoder_check, deepcenter_check

CACHE = ROOT / ".biohub/cache/public-d4-correction-v1"


@pytest.mark.skipif(not CACHE.exists(), reason="Pinned public source cache not installed")
def test_actual_two_encoder_fusion_matches_independent_group_mean():
    rows = encoder_check(CACHE)
    assert len(rows) == 6
    assert max(r["maximum_absolute_error"] for r in rows) <= 1e-12


@pytest.mark.skipif(not CACHE.exists(), reason="Pinned public source cache not installed")
def test_actual_deepcenter_group_mean_rectangular_branch_and_cache():
    rows = deepcenter_check(CACHE)
    assert len(rows) == 4
    assert max(r["maximum_absolute_error"] for r in rows) <= 1e-6
    assert all(r["cache_hit_no_new_calls"] for r in rows)
