import pytest
from research.kaggle_trajectory_evidence_v1 import evidence_name


@pytest.mark.parametrize('name',['submission.csv','trajectory-smoke/result.json','trajectory-complete/shard-0/movie-original/raw-candidates.npz'])
def test_evidence_allowed(name):assert evidence_name(name)


@pytest.mark.parametrize('name',['trajectory-site-v1/torch/x.so','trajectory-runtime-v1/model.pt','/submission.csv',
    'trajectory-smoke/../../secret','trajectory-smoke/../x','trajectory-smoke/C:x','trajectory-smoke\\x',''])
def test_libraries_and_unsafe_paths_rejected(name):assert not evidence_name(name)
