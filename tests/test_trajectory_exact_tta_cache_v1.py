import ast
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_exact_tta_cache_v1 import cached_predictor


def test_duplicate_input_identity_even_for_rectangular_views():
    for shape in ((1,2,3,4,4),(1,2,3,4,7)):
        value=np.arange(np.prod(shape)).reshape(shape)
        final=np.rot90(value,1,axes=(-2,-1)).swapaxes(-1,-2)
        assert np.array_equal(final,np.flip(value,axis=-1))


def test_original_contributions_and_inverse_transforms_remain():
    path=ROOT/'.biohub/cache/public-d4-preflight-v1/public-predictor-original.py'
    original=path.read_text();new=cached_predictor(original);ast.parse(new)
    for token in ('det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))',
                  'secondary_det_at[f].transpose(-1, -2)',
                  'det_logits[f] = det_logits[f] / _nv',
                  'secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv'):
        assert original.count(token)==new.count(token)==1
    assert new.count('_nv += 1')==original.count('_nv += 1')
    assert new.count('torch.equal(')==original.count('torch.equal(')+2
    assert new.count('model.encode(')==original.count('model.encode(')-2


def test_cache_requires_exact_pinned_layout():
    import pytest
    with pytest.raises(ValueError):cached_predictor('def unrelated(): pass')
