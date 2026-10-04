from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
V2=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v2.py'))


def test_sparse_annotations_need_not_fill_every_image_frame():
    V2['validate_sparse_times']([0,1,20,86])
    V2['validate_sparse_times'](list(range(100)))
    for values in ([],[-1],[100],[1.5]):
        with pytest.raises(ValueError):V2['validate_sparse_times'](values)


def test_truth_manifest_must_match_before_opening_geff(tmp_path):
    p=tmp_path/'train_geff_cache_manifest.json';p.write_text('{}')
    with pytest.raises(ValueError,match='manifest changed'):
        V2['verify_truth_inventory'](tmp_path,'0'*64)
