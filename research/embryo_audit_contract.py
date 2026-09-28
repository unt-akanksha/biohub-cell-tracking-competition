"""Frozen first-four target-embryo audit after a predeclared source gain.

This opens an audit, not a new tuning pool or submission authorization.
"""
import hashlib
import json

SELECTION_REPORT_SHA='db9d75ad43a9bc74d3f38f3aef48e5e617510a6abb93205a0387e726415d080f'
CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def contract(split):
    fold=split['folds'][0]
    stems=fold['initial_complete_movie_audit']
    if (len(stems)!=4 or len(set(stems))!=4 or stems!=fold['audit_order'][:4]
        or set(stems)&set(fold['train']+fold['selection'])
        or fold['held_out_embryo']!='44b6' or fold['training_embryo']!='6bba'
        or any(not s.startswith('44b6_') for s in stems)):
        raise ValueError('Frozen disjoint first-four complete target movies required')
    return dict(version=1,stems=stems,checkpoint_sha256=CHECKPOINT_SHA,
        split_sha256=SPLIT_SHA,selection_report_sha256=SELECTION_REPORT_SHA,
        use='One frozen candidate audit; not fitting, threshold tuning or submission authorization')


def verify_gate(report_bytes,split_bytes):
    if hashlib.sha256(report_bytes).hexdigest()!=SELECTION_REPORT_SHA:
        raise ValueError('Frozen successful selection report bytes required')
    if hashlib.sha256(split_bytes).hexdigest()!=SPLIT_SHA:
        raise ValueError('Original split bytes required')
    report=json.loads(report_bytes)
    if (report['decision']!='source_selection_gain_requires_embryo_audit'
        or report['authorized_for_submission'] is not False
        or report['result']['checkpoint_sha256']!=CHECKPOINT_SHA):
        raise ValueError('Verified source gain required before audit')
    return contract(json.loads(split_bytes))
