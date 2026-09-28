"""Frozen PU transfer diagnostic on already-exposed movies; never promotion."""
import hashlib
import json

REPORT_SHA = 'a75307ffc45f2302bf96db5329924835df7a4827768ada48e4a5a9c2e8bec2f5'
CHECKPOINT_SHA = 'b07f39a930855c1493e43ad16626643d6b666dc8c4dcc1426a4145cdcaff2cdf'
SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
STEMS = ['44b6_66f9292d', '44b6_40c45f5a', '44b6_3bb3690f', '44b6_0c582fdc']


def contract(split):
    fold = split['folds'][0]
    if (fold['initial_complete_movie_audit'] != STEMS or fold['audit_order'][:4] != STEMS
        or set(STEMS) & set(fold['train'] + fold['selection'])):
        raise ValueError('Only the four previously exposed target movies are allowed')
    return dict(version=1, stems=STEMS.copy(), checkpoint_sha256=CHECKPOINT_SHA,
        split_sha256=SPLIT_SHA, source_comparison_sha256=REPORT_SHA,
        diagnostic_only=True, previously_exposed=True, new_target_movies_opened=0,
        source_combined_gate_passed=False, authorized_for_submission=False,
        use='Frozen transfer diagnostic, not independent confirmation, tuning or promotion')


def verify(report_bytes, split_bytes):
    if (hashlib.sha256(report_bytes).hexdigest() != REPORT_SHA
        or hashlib.sha256(split_bytes).hexdigest() != SPLIT_SHA):
        raise ValueError('Exact failed combined comparison and original split required')
    report = json.loads(report_bytes)
    if (report['status'] != 'verified_owned_detector_source_comparison'
        or report['chosen_arm_for_further_evaluation'] is not None
        or report['authorized_for_submission'] is not False
        or report['versus_frozen_d4']['pu']['passes_predeclared_source_gate'] is not True
        or report['pu_versus_sparse']['passes_predeclared_source_gate'] is not False
        or report['scores']['pu']['checkpoint_sha256'] != CHECKPOINT_SHA):
        raise ValueError('Preserve measured baseline gain AND failed paired promotion gate')
    return contract(json.loads(split_bytes))
