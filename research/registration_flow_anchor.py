"""Image-only global anchor for backward flow; no node or graph manipulation."""
import numpy as np


def anchor(backward_um, forward_shift_um, reliability):
    flow = np.asarray(backward_um, dtype=np.float64)
    shift = np.asarray(forward_shift_um, dtype=np.float64)
    weight = float(reliability)
    if flow.ndim != 2 or flow.shape[1] != 3 or shift.shape != (3,):
        raise ValueError('Expected N-by-3 backward flow and three-axis forward shift')
    if not np.isfinite(flow).all() or not np.isfinite(shift).all() or not 0 <= weight <= 1:
        raise ValueError('Finite motion and reliability in [0,1] required')
    if not len(flow):
        return flow.copy()
    # Target -> source is the NEGATIVE of source -> target image translation.
    # Only the global offset changes: all pairwise local flow differences remain.
    return flow + weight * (-shift - np.median(flow, axis=0))
