import hashlib
from pathlib import Path
import runpy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/score-focus-source-flow.py'))
REPLAY = runpy.run_path(str(ROOT / 'scripts/replay-focus-bridge-official.py'))


def sample():
    coords = np.asarray([[0,1,2,3], [1,1,2,3]], dtype=np.float32)
    records = [dict(frame=t, sampled_nodes=int(t==1), trailing_border_extended_nodes=0,
                    policy='constant flow extension through trailing unsampled voxel centers',
                    coordinates_modified=False, nodes_deleted=False) for t in (1,2)]
    record = dict(stem='probe', processed_frames=3, processed_pairs=2, image_shape=[3,64,256,256],
                  all_nodes_covered=True, coordinate_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
                  sampler_receipts=records, probe_replayed=True)
    return record, coords, np.zeros((2,3), dtype=np.float32)


def test_complete_motion_receipt_preserves_empty_frame():
    M['verify_sampling'](*sample(), 3, 'probe')


@pytest.mark.parametrize('field,value', [('processed_pairs', 1), ('processed_frames', 2),
                                       ('all_nodes_covered', False), ('probe_replayed', False)])
def test_partial_or_false_probe_receipts_rejected(field, value):
    record, coords, flows = sample()
    record[field] = value
    with pytest.raises(ValueError, match='motion'):
        M['verify_sampling'](record, coords, flows, 3, 'probe')


def test_wrong_boundary_coverage_and_nonfinite_flow_rejected():
    record, coords, flows = sample()
    record['sampler_receipts'][0]['sampled_nodes'] = 2
    with pytest.raises(ValueError, match='coverage'):
        M['verify_sampling'](record, coords, flows, 3, 'probe')
    record, coords, flows = sample()
    flows[1, 0] = np.nan
    with pytest.raises(ValueError, match='motion'):
        M['verify_sampling'](record, coords, flows, 3, 'probe')


def test_first_frame_motion_must_remain_zero():
    record, coords, flows = sample()
    flows[0, 0] = 1
    with pytest.raises(ValueError, match='motion'):
        M['verify_sampling'](record, coords, flows, 3, 'probe')


def graph():
    payload = dict(nodes={'0': dict(t=0,z=1.,y=2.,x=3.), '1': dict(t=1,z=1.,y=2.,x=3.)},
                   edges=[dict(source_id=0,target_id=1)])
    return REPLAY['prediction_graph'](payload)


def test_actual_graph_edges_and_centroids_replayed():
    _, coords, _ = sample()
    M['verify_graph'](graph(), coords, [(0,1,.9)])
    with pytest.raises(ValueError, match='linking'):
        M['verify_graph'](graph(), coords, [])
    coords[0, 1] += .1
    with pytest.raises(ValueError, match='centroid'):
        M['verify_graph'](graph(), coords, [(0,1,.9)])
