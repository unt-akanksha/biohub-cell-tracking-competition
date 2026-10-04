import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / 'scripts/verify-focus-raw-detections.py'))


def write_fixture(root):
    (root / 'raw_detections').mkdir()
    records = []
    for stem in sorted(MODULE['STEMS']):
        path = root / 'raw_detections' / f'{stem}.npz'
        np.savez_compressed(path, coords=[[0, 1., 2., 3.]], movie_shape=[2, 8, 16, 16],
                            scale_um=[1.625, .40625, .40625])
        record = dict(stem=stem, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      movie_shape=[2, 8, 16, 16], node_count=1, frame_counts=[1, 0],
                      coordinate_source='raw instance centroids', failed_frames=0,
                      postprocessing_applied=False, ground_truth_opened=False)
        path.with_suffix('.json').write_text(json.dumps(record))
        records.append(record)
    terminal = dict(run_id='focus3d-raw-detections-v1', status='completed',
                    ground_truth_opened=False, postprocessing_applied=False,
                    submission_created=False, competition_test_data_read=False,
                    public_predictions_copied=False, metric_hack_used=False,
                    frozen_stems=sorted(MODULE['STEMS']), raw_detections=records)
    (root / 'focus3d_raw_detections_terminal.json').write_text(json.dumps(terminal))
    return terminal


def test_complete_checkpoint_is_verified_without_accuracy_claim(tmp_path):
    write_fixture(tmp_path)
    result = MODULE['verify'](tmp_path)
    assert result['status'] == 'verified_raw_detections'
    assert result['authorized_for_submission'] is False


def test_transport_corruption_is_rejected(tmp_path):
    terminal = write_fixture(tmp_path)
    (tmp_path / 'raw_detections' / (terminal['raw_detections'][0]['stem'] + '.npz')).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash mismatch'):
        MODULE['verify'](tmp_path)


def test_terminal_cannot_hide_missing_frames(tmp_path):
    terminal = write_fixture(tmp_path)
    terminal['raw_detections'][0]['frame_counts'] = [1]
    (tmp_path / 'focus3d_raw_detections_terminal.json').write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match='receipt differs'):
        MODULE['verify'](tmp_path)
