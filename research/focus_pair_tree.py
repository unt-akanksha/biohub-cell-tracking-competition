"""Bounded nonlinear residual ranking with complete candidate-group likelihood."""
import json
from pathlib import Path

import numpy as np

VERSION = '3.4.1'
ROUNDS = 100
SETTINGS = dict(tree_method='hist', device='cpu', nthread=2, max_depth=3,
                max_bin=64, eta=.05, min_child_weight=1., reg_lambda=1.,
                reg_alpha=0., max_delta_step=1., subsample=1., colsample_bytree=1.,
                seed=244691, base_score=0., disable_default_eval_metric=1)


def validate_groups(starts, sizes, chosen, present, rows):
    arrays = [np.asarray(a) for a in (starts, sizes, chosen, present)]
    starts, sizes, chosen, present = arrays
    if (any(a.dtype != np.int64 or a.ndim != 1 for a in arrays)
            or not len(starts) or any(a.shape != starts.shape for a in arrays)
            or (sizes < 1).any() or sizes.sum() != rows
            or not np.array_equal(starts, np.cumsum(np.r_[0, sizes[:-1]]))
            or (chosen < starts).any() or (chosen >= starts+sizes).any()
            or not np.array_equal(present, (chosen != starts+sizes-1).astype(np.int64))):
        raise ValueError('Complete contiguous known candidate/null groups required')
    return np.repeat(np.arange(len(starts)), sizes)


def derivatives(score, starts, sizes, chosen, present, absent_weight):
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not np.isfinite(score).all() or not np.isfinite(absent_weight) or absent_weight <= 0:
        raise ValueError('Finite scores and positive fixed class weight required')
    ids = validate_groups(starts, sizes, chosen, present, len(score))
    maxima = np.maximum.reduceat(score, starts)
    numerator = np.exp(score-maxima[ids])
    total = np.add.reduceat(numerator, starts)
    probability = numerator/total[ids]
    weight = np.where(present == 1, 1., absent_weight)
    loss = float(weight @ (maxima+np.log(total)-score[chosen]))
    gradient = probability*weight[ids]
    gradient[chosen] -= weight
    # A diagonal upper bound, not the exact coupled group Hessian:
    # D-H = sum_{i<j} w*p_i*p_j*(e_i+e_j)(e_i+e_j)^T is PSD.
    diagonal_bound = np.maximum(2*weight[ids]*probability*(1-probability), 1e-12)
    return loss, gradient, diagonal_bound


def portable_predict(model, features):
    features = np.asarray(features, dtype=np.float32)
    if (model['version'] != VERSION or model['settings'] != SETTINGS
            or features.ndim != 2 or features.shape[1] != 72
            or not np.isfinite(features).all() or len(model['trees']) != ROUNDS):
        raise ValueError('Exact finite 72D fixed-tree representation required')
    result = np.zeros(len(features), dtype=np.float32)
    for tree in model['trees']:
        pending = [(tree, np.arange(len(features)), 0)]
        while pending:
            node, indices, depth = pending.pop()
            if depth > SETTINGS['max_depth']:
                raise ValueError('Tree exceeds declared depth')
            if 'leaf' in node:
                leaf = np.float32(node['leaf'])
                if not np.isfinite(leaf) or abs(leaf) > SETTINGS['eta']*SETTINGS['max_delta_step']+1e-7:
                    raise ValueError('Finite bounded residual leaf required')
                result[indices] += leaf
                continue
            field = node['split']
            if not isinstance(field, str) or not field.startswith('f') or not field[1:].isdigit():
                raise ValueError('Original numbered feature identity required')
            column = int(field[1:])
            threshold = np.float32(node['split_condition'])
            if column >= 72 or not np.isfinite(threshold):
                raise ValueError('Valid finite feature threshold required')
            children = {child['nodeid']: child for child in node['children']}
            if len(children) != 2 or node['yes'] == node['no'] or set(children) != {node['yes'], node['no']}:
                raise ValueError('Exact two-child tree required')
            go_left = features[indices, column] < threshold
            pending.extend([(children[node['yes']], indices[go_left], depth+1),
                            (children[node['no']], indices[~go_left], depth+1)])
    if not np.isfinite(result).all():
        raise ValueError('Finite residual prediction required')
    return result


def fit(features, margin, starts, sizes, chosen, present, absent_weight, output, role, progress=None):
    if role != 'fitting':
        raise ValueError('Only fitting examples may enter residual tree training')
    import xgboost as xgb
    if xgb.__version__ != VERSION:
        raise ValueError('Pinned XGBoost version required')
    x = np.asarray(features, dtype=np.float32)
    margin = np.asarray(margin, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 72 or margin.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(margin).all()):
        raise ValueError('Complete finite features and external fixed baseline required')
    validate_groups(starts, sizes, chosen, present, len(x))
    initial, _, _ = derivatives(margin, starts, sizes, chosen, present, absent_weight)
    output = Path(output)
    output.mkdir(exist_ok=False)
    matrix = xgb.DMatrix(x, nthread=2)
    trace = []

    def objective(raw, data):
        if data is not matrix or raw.shape != margin.shape:
            raise ValueError('Never mix candidate groups across fitting matrices')
        loss, gradient, bound = derivatives(margin+raw, starts, sizes, chosen, present, absent_weight)
        trace.append(loss)
        if progress and len(trace) % 25 == 0:
            progress(dict(round=len(trace), fitting_loss=loss))
        return gradient, bound

    class Checkpoint(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if (epoch+1) % 25 == 0:
                model.save_model(output / f'partial-{epoch+1:03d}.ubj')
            return False

    native = xgb.train(SETTINGS, matrix, num_boost_round=ROUNDS, obj=objective,
                       callbacks=[Checkpoint()], verbose_eval=False)
    native.save_model(output / 'native.ubj')
    model = dict(version=VERSION, settings=SETTINGS, rounds=ROUNDS, role=role,
                 external_baseline_required=True, authorized_for_submission=False,
                 trees=[json.loads(t) for t in native.get_dump(dump_format='json')])
    expected = native.predict(matrix, output_margin=True)
    actual = portable_predict(model, x)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    final, _, _ = derivatives(margin+actual, starts, sizes, chosen, present, absent_weight)
    if not np.isfinite(final) or final >= initial or len(trace) != ROUNDS:
        raise ValueError('Full fixed tree training must improve its actual fitting loss')
    model.update(initial_loss=initial, final_loss=final, loss_trace=trace,
                 training_groups=len(starts), training_choices=len(x),
                 native_portable_max_error=float(np.max(np.abs(actual-expected))))
    target = output / 'portable.json'
    target.write_text(json.dumps(model, indent=2, allow_nan=False))
    restored = json.loads(target.read_text())
    if not np.array_equal(portable_predict(restored, x), actual):
        raise ValueError('Exact serialized prediction replay required')
    return model
