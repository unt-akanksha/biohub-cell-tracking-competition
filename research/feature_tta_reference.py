"""Bind a feature-only experiment to exact native-checkpoint detections."""
import hashlib
import json


def load_reference(root, checkpoint_sha, split_sha, movies):
    if root is None:
        raise ValueError('Native node reference required')
    path = root/'outputs/selection_manifest.json'
    manifest = json.loads(path.read_text())
    terminal = json.loads((root/'launcher_terminal.json').read_text())
    if (manifest['status'] != 'completed' or terminal['status'] != 'completed'
        or manifest['checkpoint_sha256'] != checkpoint_sha
        or manifest['manifest_sha256'] != split_sha
        or [r['stem'] for r in manifest['records']] != movies
        or manifest.get('edge_feature_tta')
        or any(manifest[k] is not False for k in ('target_audit_opened','ground_truth_opened','authorized_for_submission'))):
        raise ValueError('Exact native full-selection reference required')
    if any(r['processed_frames'] != r['image_shape'][0] for r in manifest['records']):
        raise ValueError('Incomplete reference movie')
    return dict(records={r['stem']:r for r in manifest['records']},
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def same_coordinates(candidate, native):
    import numpy as np
    candidate, native = np.asarray(candidate,dtype=float), np.asarray(native,dtype=float)
    if candidate.shape != native.shape or candidate.ndim != 2 or candidate.shape[1] != 4:
        return False
    if not np.isfinite(candidate).all() or not np.isfinite(native).all():
        return False
    def ordered(values):
        return values[np.lexsort(tuple(values[:,i] for i in (3,2,1,0)))]
    return bool(np.array_equal(ordered(candidate),ordered(native)))


def verify_coordinates(coords,path,expected_sha,tree_hash):
    import tracksdata as td
    if tree_hash(path) != expected_sha:
        raise ValueError('Native graph hash mismatch')
    graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
    native = graph.node_attrs().select(['t','z','y','x']).to_numpy()
    if not same_coordinates(coords,native):
        raise ValueError('Feature-only inference changed native detector nodes')
