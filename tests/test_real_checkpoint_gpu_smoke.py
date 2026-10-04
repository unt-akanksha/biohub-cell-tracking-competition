from pathlib import Path
import runpy
import sys
import numpy as np
import pytest

pytest.importorskip('tracksdata')
ROOT = Path(__file__).resolve().parents[1]


def test_exact_gate_on_cpu_with_actual_checkpoint_and_artificial_movie(tmp_path):
    import torch
    import zarr
    torch.set_num_threads(2)
    vendor = ROOT/'.biohub/vendor/kaggle-cell-tracking-competition'
    sys.path[:0] = [str(vendor/'scripts'),str(vendor/'src'),str(ROOT/'research')]
    checkpoint = ROOT/'.biohub/cache/kernel-outputs/independent-real-pilot-v3/independent_real_pilot/outputs/last.pt'
    if not checkpoint.is_file():
        pytest.skip('Recovered v3 checkpoint needed for integration smoke')
    stem = '6bba_b204cac7'
    group = zarr.open_group(str(tmp_path/(stem+'.zarr')), mode='w')
    group.create_array('0', data=np.zeros((3,16,64,64), dtype=np.float32), chunks=(1,16,64,64))
    group.attrs['image_statistics'] = {'quantiles':{'0.001':0.,'0.999':1.}}
    replay = runpy.run_path(str(ROOT/'scripts/replay-focus-bridge-official.py'))
    truth = replay['prediction_graph']({'nodes':{str(t):dict(t=t,z=8.,y=32.,x=32.) for t in range(3)},
        'edges':[dict(source_id=0,target_id=1),dict(source_id=1,target_id=2)]})
    truth.to_geff(tmp_path/(stem+'.geff'))
    smoke = runpy.run_path(str(ROOT/'research/real_checkpoint_gpu_smoke.py'))['run_smoke']
    result = smoke(checkpoint,tmp_path/stem,tmp_path,device='cpu')
    assert result['status'] == 'passed' and result['device'] == 'cpu'
    assert result['strict_reload'] and result['geff_round_trip']
    assert result['checkpoint_step'] == 100
    assert result['authorized_for_submission'] is False
