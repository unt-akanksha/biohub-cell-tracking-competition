"""Four fitting/four diagnostic movies for FOCUS-specific adaptation inputs."""
import hashlib
import json

SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
REPLAY=['6bba_f1fde7e0','6bba_23af9eeb']


def scope(payload):
    if hashlib.sha256(payload).hexdigest()!=SPLIT_SHA:raise ValueError('Frozen original split required')
    fold=json.loads(payload)['folds'][0]
    diagnostic=fold['train'][::5]
    fitting=[s for s in fold['train'] if s not in diagnostic]
    fit=[s for s in fitting if s not in REPLAY][:4]
    audit=[s for s in diagnostic if s not in REPLAY][:4]
    if (len(fit)!=4 or len(audit)!=4 or set(fit)&set(audit) or set(fit+audit)&set(fold['selection']+fold['audit_order'])
        or not set(fit+audit+REPLAY)<=set(fold['train'])):raise ValueError('Disjoint original training-only movie scope required')
    return dict(fitting_stems=fit,diagnostic_stems=audit,training_stems=fit+audit,replay_stems=REPLAY,
        split_sha256=SPLIT_SHA,role_assignment='Original fixed96/24 training partition, first four per role excluding replay movies',
        source_selection_opened=False,new_target_movies_opened=0,authorized_for_submission=False,
        diagnostic_caveat='Excluded from future adaptation only; original encoder/linker training included these movies')
