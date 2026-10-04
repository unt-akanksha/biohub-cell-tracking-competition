import runpy
from pathlib import Path

M = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                      'research/public_edge_supervision.py'))


def test_sparse_unmatched_is_unknown():
    f = M['classify_edge']
    assert f(1, 2, {1: 10}, {(10, 20)}, {10}, {20}) == -1
    assert f(1, 2, {1: 10, 2: -1}, {(10, 20)}, {10}, {20}) == -1


def test_negative_requires_closed_annotation_at_both_ends():
    f = M['classify_edge']
    assert f(1, 2, {1: 10, 2: 21}, {(10, 20)}, {10}, {21}) == 0
    assert f(1, 2, {1: 10, 2: 21}, {(10, 20)}, {10}, {20}) == -1
    assert f(1, 2, {1: 10, 2: 20}, {(10, 20)}, {10}, {20}) == 1


def test_mapped_pairs_deduplicates_annotation_matches():
    assert M['mapped_pairs']([(1, 2), (3, 2), (2, 4)],
                            {1: 10, 3: 10, 2: 20}) == {(10, 20)}
