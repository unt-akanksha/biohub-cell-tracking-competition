"""Fixed ensemble authorization is source validation only, never submission."""
import hashlib
import json
import math

PARENT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
PU_SHA='b07f39a930855c1493e43ad16626643d6b666dc8c4dcc1426a4145cdcaff2cdf'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def receipt(payload, split_payload):
    report=json.loads(payload); split=json.loads(split_payload)
    result=report['result']; candidate=result['candidate']; control=result['control']
    ensemble=candidate['pre_motion_patch_receipt']
    if (hashlib.sha256(split_payload).hexdigest()!=SPLIT_SHA
        or report['status']!='verified_owned_detector_ensemble_probe_not_selection'
        or result['status']!='passed_owned_detector_ensemble_functionality'
        or result['parent_sha256']!=PARENT_SHA or result['secondary_sha256']!=PU_SHA
        or result['split_sha256']!=SPLIT_SHA or result['movie']!=split['folds'][0]['train'][0]
        or result['frames']!=3 or len(result['frozen_hashes'])!=3
        or any(r['before']!=r['after'] or len(r['before'])!=64 for r in result['frozen_hashes'])
        or ensemble['weights']!=[.5,.5] or ensemble['probability_mixture'] is not True
        or ensemble['features']!='parent native unchanged' or ensemble['encode_calls']!=2
        or not math.isfinite(ensemble['maximum_mean_absolute_logit_delta'])
        or ensemble['maximum_mean_absolute_logit_delta']<=0
        or any(result[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Successful fixed training-only ensemble smoke required')
    for row in (candidate,control):
        if (row['status']!='passed' or row['frames']!=3 or row['predicted_nodes']<=0
            or row['strict_reload'] is not True or row['geff_round_trip'] is not True
            or row['standalone_image_flow'] is not True
            or row['encode_patch_receipt']['limit']!=2048
            or row['encode_patch_receipt']['maximum_nodes']>2048
            or row['encode_patch_receipt']['truncation'] is not False):
            raise ValueError('Nonempty guarded round-trip smoke required')
    for d4 in (ensemble['parent_d4'],ensemble['secondary_d4']):
        if d4['views']!=8 or d4['encode_calls']!=2:
            raise ValueError('Both detector D4 passes must execute')
    return dict(version=1,parent_sha256=PARENT_SHA,secondary_sha256=PU_SHA,
        weights=[.5,.5],probability_mixture=True,views_per_model=8,
        features='parent native unchanged',original_threshold_preserved=True,
        probe_report_sha256=hashlib.sha256(payload).hexdigest(),
        source_validation_only=True,authorized_for_submission=False)
