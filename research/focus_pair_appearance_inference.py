"""Label-free bounded inference for the frozen pair-appearance model contract."""
import numpy as np

from research.focus_candidate_ranker import candidate_arrays
from research.focus_pair_appearance import descriptors, transform

MAX_WORKING_BYTES = 1024**3


def validate_model(model):
    arm = model['arm']
    if (arm not in ('full', 'lda') or model['role'] != 'fitting' or model['ridge'] != 1.
            or model['null_logit'] != -4.5 or model['squared_correction_upper_bound'] != .5):
        raise ValueError('Exact frozen fitted appearance contract required')
    theta = np.asarray(model['theta'], float)
    if theta.shape != (72 if arm == 'full' else 9,) or not np.isfinite(theta).all() or (theta[4:7] > .5).any():
        raise ValueError('Finite exact coefficients with physical bounds required')
    transform(np.zeros((1, 64), np.float32), np.array([0], np.int64), model['projection'], arm)
    return theta


def score_blocks(packet, parameters, model, target_block=32):
    """Yield complete source-plus-null logits for each original target block.

    This function neither reads labels nor changes or filters detections. Calling
    it is not authorization to deploy a model that failed validation.
    """
    if type(target_block) is not int or not 1 <= target_block <= 64:
        raise ValueError('Target block must be an integer in1..64')
    theta = validate_model(model)
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    # Validate even an empty target frame without allocating any pair rows.
    candidate_arrays(packet, parameters, np.empty(0, np.int64))
    for first in range(0, nt, target_block):
        columns = np.arange(first, min(first + target_block, nt), dtype=np.int64)
        rows = len(columns) * (ns + 1)
        conservative_bytes = rows * 512 * 8 + (ns + nt) * 128 * 8
        if conservative_bytes > MAX_WORKING_BYTES:
            raise ValueError('Bounded inference working-set estimate exceeds1GiB; no candidate pruning')
        base = candidate_arrays(packet, parameters, columns)
        appearance = descriptors(packet, columns)
        extra = transform(appearance, base['null_rows'], model['projection'], model['arm'])
        features = np.column_stack([base['features'], extra])
        scores = (base['offset'] + features @ theta).reshape(len(columns), ns + 1)
        if not np.isfinite(scores).all() or np.any(scores[:, ns] != -4.5):
            raise ValueError('Complete finite real/null scores required')
        yield dict(target_columns=columns, scores=scores,
                   conservative_working_bytes=conservative_bytes)


def predict(packet, parameters, model, target_block=32):
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    source_ids, target_ids = (np.asarray(packet[k]) for k in ('source_indices', 'target_indices'))
    if (source_ids.dtype != np.int64 or target_ids.dtype != np.int64
            or source_ids.shape != (ns,) or target_ids.shape != (nt,)
            or len(set(source_ids)) != ns or len(set(target_ids)) != nt
            or (source_ids < 0).any() or (target_ids < 0).any()
            or np.intersect1d(source_ids, target_ids).size):
        raise ValueError('Unique disjoint original frame-global node identities required')
    selected = np.full(nt, -1, np.int64)
    probability = np.empty(nt, float)
    null_probability = np.empty(nt, float)
    coverage = np.zeros(nt, np.int64)
    peak_bound = 0
    for block in score_blocks(packet, parameters, model, target_block):
        columns, scores = block['target_columns'], block['scores']
        local = np.argmax(scores, axis=1)  # First source wins exact ties, null comes last.
        maximum = scores.max(axis=1)
        normalizer = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
        selected[columns] = np.where(local == ns, -1, local)
        probability[columns] = np.exp(scores[np.arange(len(columns)), local] - normalizer)
        null_probability[columns] = np.exp(-4.5 - normalizer)
        coverage[columns] += 1
        peak_bound = max(peak_bound, block['conservative_working_bytes'])
    if not np.all(coverage == 1):
        raise ValueError('Every original target must be predicted exactly once')
    selected_ids = np.full(nt, -1, np.int64)
    keep = selected >= 0
    selected_ids[keep] = source_ids[selected[keep]]
    return dict(target_indices=target_ids.copy(), source_indices=selected_ids,
                source_local_indices=selected, selected_probability=probability,
                null_probability=null_probability, conservative_working_bytes=peak_bound)
