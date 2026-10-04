from pathlib import Path
import runpy
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
AUDIT=runpy.run_path(str(ROOT/'scripts/audit-focus-null-difficulty.py'))


def test_exact_null_probability_and_margin():
    p=dict(source_coords=np.array([[0.,0.,0.]]),target_coords=np.array([[0.,0.,0.],[10.,0.,0.]]),backward_um=np.zeros((2,3)))
    result=AUDIT['measure'](p,np.array([0,1]),dict(mean_um=[0.,0.,0.],variance_um2=[1.,1.,1.]))
    assert result[0]['best_parent_minus_null']==4.5
    assert result[0]['null_nll']==pytest.approx(np.logaddexp(0,4.5))
    assert result[1]['best_parent_minus_null']<0
    assert AUDIT['summary'](result)['physical_correct_null']==1
