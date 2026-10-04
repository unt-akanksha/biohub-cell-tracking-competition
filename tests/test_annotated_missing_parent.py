import ast
from pathlib import Path
import numpy as np
import pytest

from research.annotated_missing_parent import missing_parent_columns, patch_epoch


def test_annotated_absent_parent_only_not_birth_or_unknown():
    gt = [[1, 0, 0], [0, 1, 0]]
    result = missing_parent_columns(gt, [0, 1, 2, -1], [[0, 0, 0]], [[0, 0, 0], [10, 0, 0]])
    np.testing.assert_array_equal(result, [False, True, False, False])


def test_nearby_unmatched_candidate_prevents_false_null():
    # Deliberately no source-match map: any nearby detection is enough.
    assert not missing_parent_columns([[1]], [0], [[4, 0, 0]], [[0, 0, 0]])[0]
    assert missing_parent_columns([[1]], [0], [[5, 0, 0]], [[0, 0, 0]])[0]
    assert not missing_parent_columns([[1]], [0], [[0, 16, 0]], [[0, 0, 0]])[0]
    assert missing_parent_columns([[1]], [0], [[0, 18, 0]], [[0, 0, 0]])[0]


def test_missing_division_parent_labels_both_daughters_and_empty_sources():
    np.testing.assert_array_equal(missing_parent_columns([[1, 1]], [0, 1], [], [[3, 3, 3]]), [True, True])


def test_parent_at_exact_radius_is_not_absent():
    assert not missing_parent_columns([[1]], [0], [[0, 16, 0]], [[0, 0, 0]], radius_um=6.5)[0]


@pytest.mark.parametrize('gt,matches,det,truth', [
    ([[1], [1]], [0], [], [[0, 0, 0], [1, 1, 1]]),
    ([[1]], [1], [], [[0, 0, 0]]),
    ([[1]], [-2], [], [[0, 0, 0]]),
    ([[1]], [0], [[np.nan, 0, 0]], [[0, 0, 0]]),
])
def test_invalid_supervision_rejected(gt, matches, det, truth):
    with pytest.raises(ValueError):
        missing_parent_columns(gt, matches, det, truth)


def test_patch_changes_only_loss_call_and_detects_drift():
    root = Path(__file__).resolve().parents[1]
    source = (root/'.biohub/vendor/kaggle-cell-tracking-competition/scripts/train_unet_transformer.py').read_text(encoding='utf-8')
    changed = patch_epoch(source)
    ast.parse(changed)
    assert changed.count('known_null=batch_missing_parent_masks(') == 1
    with pytest.raises(ValueError):
        patch_epoch(changed)
