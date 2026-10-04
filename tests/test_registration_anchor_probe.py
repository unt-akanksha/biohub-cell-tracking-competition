"""Exercise the standalone CPU module without importing the model package."""
from pathlib import Path
import runpy
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
CONTEXT=runpy.run_path(str(ROOT/'research/temporal_contrastive/transition_context.py'))


@pytest.mark.parametrize('shift',[(2,-3,4),(-2,3,-4),(0,0,0)])
def test_native_physical_sign_and_reversal(shift):
    rng=np.random.default_rng(20260910)
    first=rng.normal(size=(25,31,35)).astype(np.float32)
    second=np.roll(first,shift,axis=(0,1,2))
    estimate=CONTEXT['estimate_transition_context']
    forward=estimate(first,second,voxel_size_zyx_um=(1.625,.40625,.40625))
    backward=estimate(second,first,voxel_size_zyx_um=(1.625,.40625,.40625))
    np.testing.assert_allclose(forward.global_shift_zyx_um,np.array(shift)*[1.625,.40625,.40625])
    np.testing.assert_allclose(forward.global_shift_zyx_um,-np.array(backward.global_shift_zyx_um))


def test_probe_scope_and_cpu_metadata():
    worker=runpy.run_path(str(ROOT/'scripts/probe-focus-image-registration.py'))
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-focus-registration-probe.py'))['build']()
    assert worker['STEMS']==('6bba_f1fde7e0','6bba_23af9eeb')
    assert meta['enable_gpu'] is False and meta['enable_tpu'] is False
    assert meta['enable_internet'] is False and meta['is_private'] is True
    assert meta['competition_sources']==['biohub-cell-tracking-during-development']
    assert not meta['kernel_sources'] and not meta['model_sources']
    assert nb['metadata']['codex']['authorized_for_submission'] is False


def test_training_residual_screen_excludes_later_frames_and_divisions():
    from research.focus_residual_calibration import matched_residuals
    coords=np.array([[0,5,2,3],[1,3,2,3],[2,2,2,3],[3,1,2,3],
                     [0,5,5,5],[1,5,5,5],[1,6,5,5]],float)
    mapping={i:i+10 for i in range(len(coords)) if coords[i,0]<3}
    edges=[(10,11),(11,12),(12,13),(14,15),(14,16)]
    result=matched_residuals(coords,np.zeros((len(coords),3)),mapping,edges)
    np.testing.assert_allclose(result,[[3.25,0,0],[1.625,0,0]])
