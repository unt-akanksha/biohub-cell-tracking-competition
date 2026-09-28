"""Permit a training-fitted detector cutoff only after the full recall gate."""
import hashlib
import json
import math
import struct

SPARSE_SHA='0f441c6e8f1e649bd1549ef519af5ada43d4b01122b96934685d28cbb53b2aa7'
FULL_NOTEBOOK_SHA='a96c62e370344eb84f37f5bea4b9623112df773cfa06c0371e09e44d51fa2fc9'


def receipt(payload, split, checkpoint_sha256):
    try:
        from detector_calibration_records import calibration_scope
    except ImportError:
        from research.detector_calibration_records import calibration_scope
    report=json.loads(payload); fit=report['fit']; collection=report['collection']
    groups=calibration_scope(split,probe=False)
    if (checkpoint_sha256!=SPARSE_SHA or collection['candidate_sha256']!=SPARSE_SHA
        or report['status']!='verified_training_calibration_collection_full'
        or report['notebook_sha256']!=FULL_NOTEBOOK_SHA
        or report['full_calibration_completed'] is not True
        or report['full_diagnostic_recall_preserved'] is not True
        or any(report[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission','independent_validation'))
        or report['replayed_frames']!=18 or not 0<=report['maximum_replay_probability_delta']<=1e-6
        or collection['groups']!=groups or collection['models_unchanged'] is not True
        or fit['status']!='fitted_training_only_threshold' or fit['maximum_recall_loss']!=.005):
        raise ValueError('Verified full training-only calibration and all diagnostic safeguards required')
    threshold=fit['threshold']
    if (not math.isfinite(threshold) or not fit['baseline_threshold']<=threshold<1
        or struct.unpack('f',struct.pack('f',threshold))[0]!=threshold):
        raise ValueError('Finite representable FP32 training-fitted cutoff required')
    expected=[(group,s) for group,stems in groups.items() for s in stems]
    movies=report['per_movie']; frames=report['per_frame']
    if [(r['group'],r['stem']) for r in movies]!=expected or len(frames)!=360:
        raise ValueError('Full96/24 training movie evidence required')
    fitting=[]
    for movie in movies:
        rows=[r for r in frames if r['stem']==movie['stem']]
        if len(rows)!=3 or len({r['t'] for r in rows})!=3 or any(r['group']!=movie['group'] for r in rows):
            raise ValueError('Exactly three annotated frames per registered training movie required')
        for r in rows:
            if any(type(r[k]) is not int for k in ('annotations','parent_matched','candidate_calibrated_matched')):
                raise ValueError('Integral recall counts required')
            if not 0<=r['parent_matched']<=r['annotations'] or not 0<=r['candidate_calibrated_matched']<=r['annotations'] or r['annotations']<=0:
                raise ValueError('Invalid recall denominator/count')
        counts={k:sum(r[k] for r in rows) for k in ('annotations','parent_matched','candidate_calibrated_matched')}
        if any(movie[k]!=v for k,v in counts.items()): raise ValueError('Per-movie and per-frame counts disagree')
        if movie['group']=='diagnostic':
            if counts['candidate_calibrated_matched']<counts['parent_matched']-math.floor(.005*counts['annotations']):
                raise ValueError('A diagnostic movie failed recall preservation')
        else: fitting.extend(rows)
    n=sum(r['annotations'] for r in fitting); parent=sum(r['parent_matched'] for r in fitting)
    required=parent-math.floor(.005*n)
    if (fit['annotated_nodes']!=n or fit['parent_matched']!=parent or fit['required_matched']!=required
        or sum(r['candidate_calibrated_matched'] for r in fitting)<required):
        raise ValueError('Exact training fitting recall constraint failed')
    return dict(version=1,threshold=threshold,baseline_threshold=fit['baseline_threshold'],
        calibration_report_sha256=hashlib.sha256(payload).hexdigest(),checkpoint_sha256=checkpoint_sha256,
        fitting_movies=96,diagnostic_movies=24,scope='Training-only cutoff; complete source validation still required',
        authorized_for_submission=False)
