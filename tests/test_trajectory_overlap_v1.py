"""Scheduling may change; accepted FP32 inference mathematics may not."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research'))
from research.trajectory_portable_adapter_v1 import portable_source
from research.trajectory_runtime_v1 import verify_bundle


def test_overlap_only_changes_four_resource_controls():
    source = (ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle/run-trajectory-division-full-movie-v1.py').read_text()
    expected = portable_source(source)
    for old,new in (("OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'",
                     "OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'"),
                    ("POLARS_MAX_THREADS='2'","POLARS_MAX_THREADS='1'"),
                    ('torch.set_num_threads(2)','torch.set_num_threads(1)'),
                    ('torch.cuda.set_per_process_memory_fraction(.70)',
                     'torch.cuda.set_per_process_memory_fraction(.35)')):
        assert expected.count(old)==1
        expected=expected.replace(old,new)
    actual=(ROOT/'.biohub/cache/trajectory-overlap-v1-bundle/portable-worker.py').read_text()
    assert actual==expected
    assert 'autocast' not in actual
    assert 'torch.backends.cuda.matmul.allow_tf32 = False' in actual
    assert 'torch.backends.cudnn.allow_tf32 = False' in actual


def test_overlap_bundle_and_truth_exclusion():
    bundle=ROOT/'.biohub/cache/trajectory-overlap-v1-bundle'
    contract=verify_bundle(bundle,'cb3a8e74a63f69ac270ea5c6995c7260dcd80ea15dcbd1faefa95586681cef1c')
    assert len(contract['bundle_sha256'])==20
    assert contract['no_precision_or_model_math_changes']
    assert not contract['authorized_for_submission']


def test_whole_movie_coverage_balanced_without_duplicates():
    script=runpy.run_path(str(ROOT/'scripts/run-trajectory-overlap-v1.py'))
    movies=script['MOVIES']
    assert len(movies)==2
    assert all(len(v)==2 for v in movies)
    assert len(set(movies[0]+movies[1]))==4
    assert {v.split('_')[0] for v in movies[0]}=={'44b6','6bba'}
    assert {v.split('_')[0] for v in movies[1]}=={'44b6','6bba'}


def test_gpu_pid_query_never_mutates_foreign_processes(monkeypatch):
    script=runpy.run_path(str(ROOT/'scripts/run-trajectory-overlap-v1.py'))
    calls=[]
    class Result:
        stdout='125\n 782\n'
    def query(command,**kwargs):
        calls.append((command,kwargs))
        return Result()
    monkeypatch.setattr(script['subprocess'],'run',query)
    assert script['gpu_pids']()=={125,782}
    assert calls[0][0]==['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']
    assert calls[0][1]['timeout']==15


def test_gpu_pid_query_accepts_idle(monkeypatch):
    script=runpy.run_path(str(ROOT/'scripts/run-trajectory-overlap-v1.py'))
    class Result:
        stdout='\n'
    monkeypatch.setattr(script['subprocess'],'run',lambda *a,**kw:Result())
    assert script['gpu_pids']()==set()
