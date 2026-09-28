"""Apply independently audited source metric signs to previously ignored labels."""
import numpy as np


def relabel(original, components):
    values = np.asarray(original)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or not set(values.tolist()) <= {-1, 0, 1}:
        raise ValueError('Expected one-dimensional partial binary labels')
    result = values.copy()
    seen = set()
    for row in components:
        index, label = row['index'], row['old_label']
        if (not isinstance(index, int) or index in seen or not 0 <= index < len(result)
                or result[index] != -1 or label['label'] != -1
                or label.get('known_children', 0) < 1 or label.get('unknown_children', 0)
                or label.get('ambiguous_children', 0)):
            raise ValueError('Only unique fully-known previously ignored components may change')
        seen.add(index)
        delta = float(row['delta_score'])
        if not np.isfinite(delta):
            raise ValueError('Non-finite official counterfactual delta')
        result[index] = 1 if delta > 1e-12 else 0 if delta < -1e-12 else -1
    return result
