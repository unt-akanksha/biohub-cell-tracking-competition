import json
from pathlib import Path
import numpy as np
import pytest
from research.detector_calibration_records import calibration_scope,select_frames,temporal_context,annotated_confidences


def test_training_split_and_probe_are_disjoint_and_fixed():
    split=json.loads((Path(__file__).resolve().parents[1]/'research/independent_real_baseline_v1_split.json').read_text())
    full=calibration_scope(split,False); probe=calibration_scope(split)
    assert len(full['fitting'])==96 and len(full['diagnostic'])==24
    assert not set(full['fitting'])&set(full['diagnostic'])
    assert probe==dict(fitting=full['fitting'][:4],diagnostic=full['diagnostic'][:2])
    split['folds'][0]['train'][0]=split['folds'][0]['selection'][0]
    with pytest.raises(ValueError): calibration_scope(split)


def test_annotation_frame_choice_ignores_iteration_order_and_duplicates():
    chosen=select_frames('movie',list(range(100)),100)
    assert len(chosen)==3 and chosen==sorted(chosen)
    assert chosen==select_frames('movie',list(reversed(range(100)))+[0],100)
    assert temporal_context(0,100)==((0,1),0)
    assert temporal_context(99,100)==((98,99),1)
    with pytest.raises(ValueError): temporal_context(100,100)
    with pytest.raises(ValueError): select_frames('movie',[0,1],100)


def test_official_matching_never_counts_two_peaks_for_one_annotation():
    coords=np.array([[1,0,0,0],[1,0,0,1],[1,0,50,50]])
    truth=np.array([[1,0,0,0],[1,0,100,100]])
    result=annotated_confidences(coords,[.8,.9,.95],truth)
    assert result.shape==(2,) and np.sum(result>0)==1 and result[1]==0
    assert result[0]==np.float32(.8)


def test_empty_predictions_keep_every_annotation():
    result=annotated_confidences(np.empty((0,4)),[],np.array([[0,0,0,0],[1,0,0,0]]))
    assert np.array_equal(result,[0.,0.])


def test_official_matching_respects_time_and_physical_scale():
    coords=np.array([[0,0,0,0],[1,5,0,0]])
    truth=np.array([[1,0,0,0]])
    # Wrong time or 8.125um Z displacement cannot match at 7um.
    assert annotated_confidences(coords,[.9,.9],truth)[0]==0
