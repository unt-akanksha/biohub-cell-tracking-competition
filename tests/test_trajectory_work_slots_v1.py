"""Whole-movie ownership and two-physical-GPU limits remain mandatory."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from trajectory_work_slots_v1 import build_movie_slots,validate_slot_outputs
from submission_sharding import MovieShard


@pytest.mark.parametrize('count',[2,3,4,8,199])
def test_unique_full_coverage(count):
    movies=[f'movie_{i}' for i in range(count)]
    weights={m:float(i+1) for i,m in enumerate(movies)}
    slots=build_movie_slots(movies,('GPU-a','GPU-b'),movie_weights=weights)
    observed={s.shard_index:list(s.movie_ids) for s in slots}
    assert validate_slot_outputs(slots,observed)==tuple(sorted(movies))
    assert len(slots)<=4
    assert {s.cuda_token for s in slots}=={'GPU-a','GPU-b'}
    assert all(sum(s.cuda_token==token for s in slots)<=2 for token in ('GPU-a','GPU-b'))


def test_missing_or_duplicate_outputs_rejected():
    slots=build_movie_slots(['a','b','c','d'],('0','1'),movie_weights=dict(a=1,b=1,c=1,d=1))
    observed={s.shard_index:list(s.movie_ids) for s in slots}
    missing=dict(observed);missing.pop(0)
    with pytest.raises(ValueError):validate_slot_outputs(slots,missing)
    duplicated=dict(observed);duplicated[0]=observed[0]*2
    with pytest.raises(ValueError):validate_slot_outputs(slots,duplicated)


def test_third_gpu_rejected():
    slots=(MovieShard(0,3,'0',('a',)),MovieShard(1,3,'1',('b',)),MovieShard(2,3,'2',('c',)))
    with pytest.raises(ValueError):validate_slot_outputs(slots,{0:['a'],1:['b'],2:['c']})


def test_cpu_smoke_can_materialize_offline_overlay(tmp_path):
    import ast,hashlib,json,shutil
    staging=ROOT/'.biohub/staging/biohub-trajectory-overlap-acceptance-v1'
    proof=json.loads((ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-build.json').read_text())
    overlays=json.loads((staging/'overlays.json').read_text())
    contract=json.loads((staging/'derived-contract.json').read_text())
    assert hashlib.sha256((staging/'derived-contract.json').read_bytes()).hexdigest()==proof['contract_sha256']
    for name,text in overlays.items():
        dest=tmp_path/name;dest.write_bytes(text.encode('utf-8'))
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==contract['bundle_sha256'][name]
        if name.endswith('.py'):ast.parse(text)
    notebook=json.loads((staging/'trajectory-overlap-acceptance.ipynb').read_text())
    code=''.join(notebook['cells'][1]['source']);ast.parse(code)
    assert code.index('verify_bundle(bundle, CONTRACT_SHA256)')<code.index('_overlays =')
    assert "verify_bundle(bundle,CONTRACT_SHA256)" in code
    assert 'autocast' not in overlays['portable-worker.py']
