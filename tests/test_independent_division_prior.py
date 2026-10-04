from pathlib import Path
import runpy

import numpy as np
import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/analyze-independent-division-prior.py'))


def test_null_geometry_boundary_and_zero_motion():
    sigma = np.sqrt(MODULE['VARIANCE'][0])
    result = MODULE['summarize']([[0,0,0], [4*sigma,0,0]])
    assert result['edges'] == 2 and result['prior_not_above_null'] == 1
    assert result['fraction_prior_not_above_null'] == .5
    assert result['residual_required_to_exceed_null']['p50'] == pytest.approx(1.75)


def test_empty_and_nonfinite_displacements_rejected():
    for values in ([], [[float('nan'),0,0]]):
        with pytest.raises(ValueError):
            MODULE['summarize'](values)
