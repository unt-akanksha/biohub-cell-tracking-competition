"""Label-free previous-image-motion features for complete candidate groups.

This does not extrapolate ground-truth or previously accepted tracking links.
It joins the preceding packet's image-predicted target flow by original node ID.
No model, candidate selection, or submission authorization is provided here.
"""
import numpy as np

from research.focus_candidate_ranker import SCALE, candidate_arrays

FEATURES = ('history_available', 'history_delta_z', 'history_delta_y',
            'history_delta_x', 'history_delta_z_squared',
            'history_delta_y_squared', 'history_delta_x_squared')
MAX_WORKING_BYTES = 256 * 1024**2


def _ids(packet, side):
    coords = np.asarray(packet[side+'_coords'])
    ids = np.asarray(packet[side+'_indices'])
    if (ids.dtype != np.int64 or ids.shape != (len(coords),)
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Unique original int64 node IDs required')
    return ids


def previous_source_motion(packet, previous):
    """Return source-aligned backward image flow and an explicit boundary mask."""
    frame = np.asarray(packet['source_frame'])
    if frame.shape != () or not np.issubdtype(frame.dtype, np.integer) or int(frame) < 0:
        raise ValueError('Nonnegative scalar integer source frame required')
    source = np.asarray(packet['source_coords'])
    ids = _ids(packet, 'source')
    if source.shape != (len(ids), 3) or not np.isfinite(source).all():
        raise ValueError('Finite original source geometry required')
    if previous is None:
        if int(frame) != 0:
            raise ValueError('Missing previous packet is allowed only at movie start')
        return np.zeros((len(ids), 3), float), np.zeros(len(ids), bool)
    old_frame = np.asarray(previous['source_frame'])
    if (old_frame.shape != () or not np.issubdtype(old_frame.dtype, np.integer)
            or int(old_frame) < 0 or int(old_frame)+1 != int(frame)):
        raise ValueError('Exactly preceding transition required')
    old_ids = _ids(previous, 'target')
    old_coords = np.asarray(previous['target_coords'])
    flow = np.asarray(previous['backward_um'], float)
    if (old_coords.shape != (len(old_ids), 3) or flow.shape != (len(old_ids), 3)
            or not np.isfinite(old_coords).all() or not np.isfinite(flow).all()
            or not np.array_equal(np.sort(ids), np.sort(old_ids))):
        raise ValueError('Previous targets must cover every current source exactly')
    order = np.argsort(old_ids)
    aligned = order[np.searchsorted(old_ids[order], ids)]
    if not np.array_equal(source, old_coords[aligned]):
        raise ValueError('Same original node IDs must have exactly the same geometry')
    return flow[aligned].copy(), np.ones(len(ids), bool)


def candidate_history(packet, previous, parameters, columns=None):
    """Seven additional features, target-major with every source then null.

    Flow is backward: source(t)-target(t+1) should agree with the image-predicted
    backward motion at source(t), under a constant-velocity hypothesis. This
    is a feature, not a hard constraint; the learned model may ignore it.
    Scaling and bias use the existing frozen physical parameters. Null features
    and all initial-frame history features are zero. Labels are never accessed.
    """
    # Reuse the original finite geometry/feature/parameter validation, without
    # allocating pair rows or reading labels.
    candidate_arrays(packet, parameters, np.empty(0, np.int64))
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    source_ids, target_ids = _ids(packet, 'source'), _ids(packet, 'target')
    if np.intersect1d(source_ids, target_ids).size:
        raise ValueError('Adjacent frames require disjoint original node IDs')
    if columns is None:
        columns = np.arange(nt, dtype=np.int64)
    columns = np.asarray(columns)
    if (columns.dtype != np.int64 or columns.ndim != 1
            or len(np.unique(columns)) != len(columns)
            or np.any(columns < 0) or np.any(columns >= nt)):
        raise ValueError('Unique original target columns required')
    estimate = len(columns)*(ns+1)*7*8*3 + (ns+nt)*64*8
    if estimate > MAX_WORKING_BYTES:
        raise ValueError('History allocation exceeds256MiB; use target blocks, not pruning')
    flow, available = previous_source_motion(packet, previous)
    features = np.zeros((len(columns), ns+1, len(FEATURES)), float)
    if np.any(available):
        source, target = np.asarray(packet['source_coords'], float), np.asarray(packet['target_coords'], float)
        mean, variance = np.asarray(parameters['mean_um']), np.asarray(parameters['variance_um2'])
        delta = ((source[None]-target[columns, None])*SCALE-flow[None]-mean)/np.sqrt(variance)
        features[:, :ns, 0] = available
        features[:, :ns, 1:4] = delta
        features[:, :ns, 4:7] = delta**2
    if not np.isfinite(features).all():
        raise ValueError('Finite complete temporal-history features required')
    return dict(features=features.reshape(-1, len(FEATURES)), target_columns=columns.copy(),
                source_history_available=available, conservative_working_bytes=estimate)
