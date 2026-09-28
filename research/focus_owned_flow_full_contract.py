"""Frozen four-movie diagnostic after an actual cached-centroid CUDA smoke."""
import hashlib
import json

PROBE_SHA='645a7613cf8fc69e5fc8127d06f6a3ae2e2994d97dddf85e5e402a412d85ad05'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
RAW_SHA='0609934b1e2a40473763acf521cfcf7120e418f5857c24d6f28c0e662638bd41'
STEMS=['44b6_81c256f0','44b6_24264f12','6bba_f1fde7e0','6bba_23af9eeb']


def receipt(probe_payload,split_payload):
    if hashlib.sha256(probe_payload).hexdigest()!=PROBE_SHA or hashlib.sha256(split_payload).hexdigest()!=SPLIT_SHA:
        raise ValueError('Exact completed GPU/CPU-replayed probe and split required')
    probe=json.loads(probe_payload); r=probe['result']
    if (probe['status']!='verified_cached_focus_owned_flow_probe_not_accuracy'
        or probe['terminal']['status']!='completed'
        or any(probe['cpu_replay'][key] is not True for key in ('raw_coordinates_exact','static_control_replayed','flow_links_replayed'))
        or r['raw_terminal_sha256']!=RAW_SHA or probe['authorized_for_submission'] is not False):
        raise ValueError('Verified unchanged-centroid graph replay required')
    fold=json.loads(split_payload)['folds'][0]
    if any(s not in fold['train'] for s in STEMS[2:]) or set(STEMS[:2]) & set(fold['train']+fold['selection']+fold['audit_order']):
        raise ValueError('Preserve the old exposed diagnostic movies, not new audit movies')
    return dict(version=1,stems=STEMS.copy(),probe_report_sha256=PROBE_SHA,split_sha256=SPLIT_SHA,
        raw_terminal_sha256=RAW_SHA,flow_checkpoint_sha256=r['checkpoint_sha256'],
        flow_tensor_sha256=r['frozen_flow_tensor_sha256'],probe_motion_sha256=r['sample_sha256'],
        probe_stem=r['movie'],flow_views=1,detector_inference=False,public_linker_loaded=False,
        coordinates='exact raw centroids',sampling='constant flow extension through trailing unsampled voxel centers',
        null_logit=-4.5,posterior_threshold=.5,max_parents=1,max_children=2,
        scope='Four previously exposed diagnostic movies; two flow-training movies and two other-embryo movies',
        new_target_movies_opened=0,independent_confirmation=False,authorized_for_submission=False)
