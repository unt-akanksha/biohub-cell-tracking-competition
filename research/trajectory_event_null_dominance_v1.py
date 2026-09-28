"""Experimental exact event pruning by equivalent birth/death explanations.

Replacing a continuation by its parent's death and child's birth, or a fork by
death plus two births, covers exactly the same constraint rows. Strictly better
allowed replacements exclude that option from every optimum. No fitted weights
or annotation-dependent constraints are introduced. Not installed in live jobs.
"""
import numpy as np
from research.trajectory_event_vectorized_dominance_v1 import allowed_options as fork_pruning


def allowed_options(case, scores, allowed=None):
    scores = np.asarray(scores, np.float64)
    retained, report = fork_pruning(case, scores, allowed)
    options = np.asarray(case['options'])
    births = np.full(int(case['nchildren']), -1, dtype=np.int64)
    deaths = np.full(int(case['nparents']), -1, dtype=np.int64)
    birth_rows = np.flatnonzero(retained & (options[:, 0] < 0))
    death_rows = np.flatnonzero(retained & (options[:, 0] >= 0) & (options[:, 1] < 0))
    births[options[birth_rows, 1]] = birth_rows
    deaths[options[death_rows, 0]] = death_rows
    removed = {}
    for is_fork, key in ((False, 'null_dominated_continuations'), (True, 'null_dominated_forks')):
        candidates = np.flatnonzero(retained & (options[:, 0] >= 0) & (options[:, 1] >= 0)
                                    & ((options[:, 2] >= 0) == is_fork))
        p, a, b = options[candidates].T
        first, second = deaths[p], births[a]
        eligible = (first >= 0) & (second >= 0)
        if is_fork:
            third = births[b]
            eligible &= third >= 0
        local = np.flatnonzero(eligible)
        with np.errstate(over='ignore', invalid='ignore'):
            replacement = scores[first[local]] + scores[second[local]]
            if is_fork:
                replacement += scores[third[local]]
            current = scores[candidates[local]]
            tolerance = 1e-10 * np.maximum(1., np.maximum(np.abs(replacement), np.abs(current)))
            dominated = np.isfinite(replacement) & (replacement > current + tolerance)
        rejected = candidates[local[dominated]]
        retained[rejected] = False
        removed[key] = len(rejected)
    report.update(removed)
    report['allowed_after'] = int(retained.sum())
    return retained, report
