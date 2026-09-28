"""Resume a preserved fixed-tree prefix; no parameter or candidate changes."""
import json
from pathlib import Path

import numpy as np

from research.focus_pair_tree import SETTINGS, VERSION, ROUNDS, derivatives, portable_predict, validate_groups


def resume(features, margin, starts, sizes, chosen, present, absent_weight, checkpoint, output, role, progress=None):
    if role != 'fitting':
        raise ValueError('Only fitting data may resume a residual model')
    import xgboost as xgb
    if xgb.__version__ != VERSION:
        raise ValueError('Pinned original tree library required')
    x, margin = np.asarray(features, np.float32), np.asarray(margin, np.float64)
    if x.shape != (len(margin), 72) or not np.isfinite(x).all() or not np.isfinite(margin).all():
        raise ValueError('Complete finite original features and baseline required')
    validate_groups(starts, sizes, chosen, present, len(x))
    output = Path(output)
    output.mkdir(exist_ok=False)
    matrix = xgb.DMatrix(x, nthread=2)
    prefix = xgb.Booster(model_file=checkpoint)
    if prefix.num_boosted_rounds() != 75:
        raise ValueError('Exactly the preserved75-tree checkpoint required')
    # Zero padding is solely for replay through the frozen100-tree evaluator.
    # The native prefix remains75trees; no padded model is persisted/deployed.
    prefix_model = dict(version=VERSION, settings=SETTINGS,
        trees=[json.loads(t) for t in prefix.get_dump(dump_format='json')]+[dict(nodeid=0, leaf=0.) for _ in range(25)])
    expected = prefix.predict(matrix, output_margin=True)
    actual = portable_predict(prefix_model, x)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    args = (starts, sizes, chosen, present, absent_weight)
    initial = derivatives(margin, *args)[0]
    resumed_initial = derivatives(margin+actual, *args)[0]
    trace = []

    def objective(raw, data):
        if data is not matrix or raw.shape != margin.shape:
            raise ValueError('Exact full candidate-group matrix required')
        loss, gradient, diagonal = derivatives(margin+raw, *args)
        trace.append(loss)
        if progress and len(trace) % 5 == 0:
            progress(dict(round=75+len(trace), fitting_loss=loss))
        return gradient, diagonal

    native = xgb.train(SETTINGS, matrix, num_boost_round=25, obj=objective,
                       xgb_model=prefix, verbose_eval=False)
    native.save_model(output/'native.ubj')
    model = dict(version=VERSION, settings=SETTINGS, rounds=ROUNDS, role=role,
                 external_baseline_required=True, authorized_for_submission=False,
                 recovered_prefix_rounds=75, loss_trace_start_round=76, loss_trace=trace,
                 trees=[json.loads(t) for t in native.get_dump(dump_format='json')])
    expected = native.predict(matrix, output_margin=True)
    actual = portable_predict(model, x)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    final = derivatives(margin+actual, *args)[0]
    if len(trace) != 25 or final >= resumed_initial or final >= initial:
        raise ValueError('Complete remaining25 rounds must improve actual fitting loss')
    model.update(initial_loss=initial, prefix_loss=resumed_initial, final_loss=final,
                 training_groups=len(starts), training_choices=len(x),
                 native_portable_max_error=float(np.max(np.abs(expected-actual))))
    target = output/'portable.json'
    target.write_text(json.dumps(model, indent=2, allow_nan=False))
    if not np.array_equal(portable_predict(json.loads(target.read_text()), x), actual):
        raise ValueError('Serialized resumed model changed predictions')
    return model
