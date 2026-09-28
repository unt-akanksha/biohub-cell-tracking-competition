"""Training-only, recall-constrained confidence calibration primitive.

Inputs are one-to-one matched annotation records, not all image peaks. Missing
annotations remain in the denominator. No negative labels are assigned to
unannotated detections and no desired prediction count is used.
"""
import math
import numpy as np


def fit_threshold(parent_matched, candidate_probabilities, *, baseline_threshold,
                  maximum_recall_loss=.005):
    parent=np.asarray(parent_matched)
    probabilities=np.asarray(candidate_probabilities,dtype=np.float32)
    base=np.float32(baseline_threshold)
    if (parent.ndim!=1 or parent.dtype!=np.bool_ or probabilities.shape!=parent.shape or not len(parent)
        or not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any()
        or not np.isfinite(base) or not 0<base<1
        or not math.isfinite(maximum_recall_loss) or not 0<=maximum_recall_loss<=.005):
        raise ValueError('Aligned annotated matches, FP32 probabilities and bounded recall loss required')
    # Unmatched candidate annotations have probability0, not omitted rows.
    if np.any((probabilities>0)&(probabilities<=base)):
        raise ValueError('Candidate records must match the exact baseline detector cutoff')
    count=len(parent); matched=int(parent.sum())
    required=matched-math.floor(maximum_recall_loss*count)
    if required<=0: raise ValueError('Insufficient parent annotation coverage for calibration')
    observed=int(np.sum(probabilities>base))
    shared=dict(annotated_nodes=count,parent_matched=matched,candidate_baseline_matched=observed,
        required_matched=required,parent_recall=matched/count,candidate_baseline_recall=observed/count,
        baseline_threshold=float(base),maximum_recall_loss=maximum_recall_loss,
        authorized_for_submission=False,independent_recall_guarantee=False)
    if observed<required:
        return dict(shared,status='baseline_recall_requirement_unmet',threshold=None)
    boundary=np.sort(probabilities)[-required]
    # Official inference is strict `sigmoid(logits) > threshold` in FP32.
    # An FP64 predecessor rounds back to the tie when converted to FP32.
    threshold=max(base,np.nextafter(boundary,np.float32(-np.inf)))
    retained=int(np.sum(probabilities>threshold))
    if retained<required or not base<=threshold<1:
        raise ValueError('Strict FP32 comparator violates annotated recall constraint')
    return dict(shared,status='fitted_training_only_threshold',threshold=float(threshold),
        candidate_retained_matched=retained,candidate_retained_recall=retained/count,
        scope='Training annotations only; independent complete-movie validation still required')
