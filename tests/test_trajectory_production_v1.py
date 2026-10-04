import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.submission_sharding import validate_submission_kernel_metadata


def test_production_only_switches_accepted_mode():
    base=ROOT/'.biohub/staging/biohub-trajectory-motion-acceptance-v3/trajectory-motion-acceptance.ipynb'
    out=ROOT/'.biohub/staging/biohub-trajectory-motion-candidate-v1/trajectory-motion-candidate.ipynb'
    original=''.join(json.loads(base.read_text())['cells'][1]['source'])
    code=''.join(json.loads(out.read_text())['cells'][1]['source'])
    assert original.count("RUN_MODE = 'acceptance'")==1
    assert code==original.replace("RUN_MODE = 'acceptance'","RUN_MODE = 'production'",1)
    ast.parse(code)
    assert "else sorted(p.stem for p in data_root.glob('*.zarr'))" in code


def test_production_quality_proof_and_explicit_resource_policy():
    receipt=json.loads((ROOT/'reports/experiments/trajectory-production-v1-build.json').read_text())
    proof=ROOT/'reports/experiments/trajectory-kaggle-acceptance-v2-result.json'
    assert sha(proof)==receipt['acceptance_sha256']
    assert json.loads(proof.read_text())['comparison']['diagnostic_gate_passed']
    assert receipt['runtime_budget_policy_revision']==2
    assert receipt['runtime_projection_hours']<=8
    assert receipt['observed_runtime_headroom_hours']>=2
    assert receipt['inference_watchdog_seconds']==36000
    assert receipt['notebook_wall_cap_seconds']==43200
    assert not receipt['submission_performed']


def test_production_private_offline_two_gpu_metadata():
    folder=ROOT/'.biohub/staging/biohub-trajectory-motion-candidate-v1'
    metadata=json.loads((folder/'kernel-metadata.json').read_text())
    validate_submission_kernel_metadata(metadata)
    assert metadata['is_private']
    assert not metadata['enable_internet']
    assert metadata['competition_sources']==['biohub-cell-tracking-during-development']
