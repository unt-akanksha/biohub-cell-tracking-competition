"""Inference never gets an annotation-derived candidate mask."""
import numpy as np


def predict(matrix, groups, weights):
    scores = matrix.astype(np.float64) @ weights
    result = np.full(len(groups['children']), -1, np.int64)
    for i, (a,b) in enumerate(zip(groups['offsets'][:-1], groups['offsets'][1:])):
        if b > a:
            result[i] = groups['parents'][a + np.argmax(scores[a:b])]
    return result


def evaluate(predictions, groups, targets):
    known = targets >= 0
    correct = predictions[known] == targets[known]
    current = groups['current'][known] == targets[known]
    initial = groups['neural'][known] == targets[known]
    return dict(queries=int(known.sum()), learned_correct=int(correct.sum()),
                current_correct=int(current.sum()), initial_correct=int(initial.sum()),
                learned_repairs=int((correct & ~current).sum()),
                learned_breaks=int((~correct & current).sum()),
                all_queries=len(targets), changed_current_choices=int((predictions != groups['current']).sum()),
                annotation_mask_used_for_inference=False)
