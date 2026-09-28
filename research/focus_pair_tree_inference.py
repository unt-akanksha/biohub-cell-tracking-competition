"""Complete label-free pair ranking with a checked native CPU tree predictor."""
import json

import numpy as np

from research.focus_candidate_ranker import candidate_arrays
from research.focus_pair_appearance import descriptors, transform
from research.focus_pair_appearance_inference import validate_model, MAX_WORKING_BYTES
from research.focus_pair_tree import portable_predict, VERSION


class NativeResidual:
    def __init__(self, model, native_path):
        portable_predict(model, np.empty((0, 72), np.float32))
        if model['role'] != 'fitting' or model['external_baseline_required'] is not True:
            raise ValueError('Original externally based fitting model required')
        import xgboost as xgb
        if xgb.__version__ != VERSION:
            raise ValueError('Pinned original native tree library required')
        self.native = xgb.Booster(params=dict(nthread=2, device='cpu'), model_file=native_path)
        if [json.loads(t) for t in self.native.get_dump(dump_format='json')] != model['trees']:
            raise ValueError('Native model must contain the exact portable100 trees')

    def predict(self, features):
        features = np.asarray(features, np.float32, order='C')
        if features.ndim != 2 or features.shape[1] != 72 or not np.isfinite(features).all():
            raise ValueError('Finite original72D tree features required')
        if len(features) == 0:
            return np.empty(0, np.float32)
        values = self.native.inplace_predict(features, predict_type='margin')
        if values.shape != (len(features),) or not np.isfinite(values).all():
            raise ValueError('One finite residual per original candidate required')
        return values


def score_blocks(packet, parameters, baseline, residual, target_block=32):
    if type(target_block) is not int or not 1 <= target_block <= 64:
        raise ValueError('Integer target block in1..64 required')
    theta = validate_model(baseline)
    if baseline['arm'] != 'full':
        raise ValueError('The nonlinear correction requires the full72D baseline')
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    candidate_arrays(packet, parameters, np.empty(0, np.int64))
    # Validate the native/portable predictor even for empty target frames.
    residual.predict(np.empty((0, 72), np.float32))
    for first in range(0, nt, target_block):
        columns = np.arange(first, min(first+target_block, nt), dtype=np.int64)
        rows = len(columns)*(ns+1)
        conservative_bytes = rows*512*8+(ns+nt)*128*8
        if conservative_bytes > MAX_WORKING_BYTES:
            raise ValueError('Complete-block memory estimate exceeds1GiB; never prune candidates')
        base = candidate_arrays(packet, parameters, columns)
        extra = transform(descriptors(packet, columns), base['null_rows'], baseline['projection'], 'full')
        features = np.column_stack([base['features'], extra])
        correction = np.asarray(residual.predict(features), np.float64)
        if correction.shape != (rows,) or not np.isfinite(correction).all():
            raise ValueError('Complete finite residual coverage required')
        scores = (base['offset']+features @ theta+correction).reshape(len(columns), ns+1)
        null_correction = correction[base['null_rows']]
        if not np.all(null_correction == null_correction[0]):
            raise ValueError('Identical null features must yield identical residuals')
        scores -= null_correction[:, None]
        if not np.isfinite(scores).all() or np.any(scores[:, ns] != -4.5):
            raise ValueError('Unchanged null reference and finite complete scores required')
        yield dict(target_columns=columns, scores=scores, conservative_working_bytes=conservative_bytes)


def predict(packet, parameters, baseline, residual, target_block=32):
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    source_ids, target_ids = (np.asarray(packet[k]) for k in ('source_indices', 'target_indices'))
    if (source_ids.dtype != np.int64 or target_ids.dtype != np.int64
            or source_ids.shape != (ns,) or target_ids.shape != (nt,)
            or len(set(source_ids)) != ns or len(set(target_ids)) != nt
            or (source_ids < 0).any() or (target_ids < 0).any()
            or np.intersect1d(source_ids, target_ids).size):
        raise ValueError('Unique disjoint original node identities required')
    selected = np.full(nt, -1, np.int64)
    probability, null_probability = np.empty(nt), np.empty(nt)
    coverage, maximum_bytes = np.zeros(nt, np.int64), 0
    for block in score_blocks(packet, parameters, baseline, residual, target_block):
        columns, scores = block['target_columns'], block['scores']
        choice = scores.argmax(axis=1)
        maximum = scores.max(axis=1)
        normalizer = maximum+np.log(np.exp(scores-maximum[:, None]).sum(axis=1))
        selected[columns] = np.where(choice == ns, -1, choice)
        probability[columns] = np.exp(scores[np.arange(len(columns)), choice]-normalizer)
        null_probability[columns] = np.exp(-4.5-normalizer)
        coverage[columns] += 1
        maximum_bytes = max(maximum_bytes, block['conservative_working_bytes'])
    if not np.all(coverage == 1):
        raise ValueError('Every original target must be predicted exactly once')
    parents = np.full(nt, -1, np.int64)
    real = selected >= 0
    parents[real] = source_ids[selected[real]]
    return dict(target_indices=target_ids.copy(), source_indices=parents, source_local_indices=selected,
                selected_probability=probability, null_probability=null_probability,
                conservative_working_bytes=maximum_bytes)
