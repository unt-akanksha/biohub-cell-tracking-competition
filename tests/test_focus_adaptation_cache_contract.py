import json
from pathlib import Path
import pytest
from research.focus_adaptation_cache_contract import scope,REPLAY

ROOT=Path(__file__).resolve().parents[1]


def test_original_training_only_disjoint_movie_roles():
    data=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    result=scope(data);fold=json.loads(data)['folds'][0]
    assert len(result['fitting_stems'])==len(result['diagnostic_stems'])==4
    assert not set(result['training_stems'])&set(REPLAY+fold['selection']+fold['audit_order'])
    assert set(result['diagnostic_stems'])<=set(fold['train'][::5])
    assert set(result['fitting_stems'])<=set(fold['train'])-set(fold['train'][::5])
    assert result['authorized_for_submission'] is False


def test_changed_split_rejected():
    with pytest.raises(ValueError):scope(b'{}')
