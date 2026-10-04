"""Check recovered raw detector checkpoints before any linker launch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_linker_runtime import load_raw_cache

STEMS = {'44b6_81c256f0', '44b6_24264f12', '6bba_f1fde7e0', '6bba_23af9eeb'}


def verify(root):
    terminal_path = root / 'focus3d_raw_detections_terminal.json'
    terminal = json.loads(terminal_path.read_text())
    if terminal['run_id'] != 'focus3d-raw-detections-v1' or terminal['status'] != 'completed':
        raise ValueError('raw detector run is not complete')
    for key in ('ground_truth_opened', 'postprocessing_applied', 'submission_created',
                'competition_test_data_read', 'public_predictions_copied', 'metric_hack_used'):
        if terminal.get(key) is not False:
            raise ValueError(f'raw detector contract failed: {key}')
    records = terminal['raw_detections']
    if len(records) != 4 or {r['stem'] for r in records} != STEMS or set(terminal['frozen_stems']) != STEMS:
        raise ValueError('raw detector movie coverage changed')
    for record in records:
        stem = record['stem']
        if json.loads((root / 'raw_detections' / f'{stem}.json').read_text()) != record:
            raise ValueError('per-movie receipt differs from terminal')
        if record['failed_frames'] != 0 or record['postprocessing_applied'] is not False:
            raise ValueError('incomplete or processed raw detections')
        if record['coordinate_source'] != 'raw instance centroids':
            raise ValueError('raw centroid provenance required')
        shape = record['movie_shape']
        cache = load_raw_cache(root / 'raw_detections', stem, record['sha256'], shape)
        if record['frame_counts'] != cache.manifest()['frame_counts'] or record['node_count'] != len(cache.node_ids):
            raise ValueError('raw detection counts differ from persisted coordinates')
        with np.load(root / 'raw_detections' / f'{stem}.npz', allow_pickle=False) as data:
            if not np.array_equal(data['scale_um'], [1.625, .40625, .40625]):
                raise ValueError('unexpected physical scale')
    return {'status': 'verified_raw_detections',
            'terminal_sha256': hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
            'movies': records, 'authorized_for_submission': False,
            'verification_scope': 'checkpoint transport, coordinates, coverage and provenance; no accuracy claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
