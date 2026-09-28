"""Eight additional fitting movies selected by frozen ordering, not outcomes."""
import json
from research.focus_adaptation_cache_contract import scope as original_scope


def scope(payload):
    original=original_scope(payload);fold=json.loads(payload)['folds'][0]
    excluded=set(fold['train'][::5]+original['fitting_stems']+original['replay_stems'])
    new=[s for s in fold['train'] if s not in excluded][:8]
    if len(new)!=8 or set(new)&set(fold['selection']+fold['audit_order']+original['diagnostic_stems']):
        raise ValueError('Eight untouched original-fitting movies required')
    return dict(fitting_stems=new,training_stems=new,replay_stems=original['replay_stems'],
        previous_fitting_stems=original['fitting_stems'],unchanged_diagnostic_stems=original['diagnostic_stems'],
        split_sha256=original['split_sha256'],source_selection_opened=False,new_target_movies_opened=0,
        authorized_for_submission=False,role_assignment='Next eight original96 fitting movies, excluding cached/replay movies; no labels or scores used',
        diagnostic_caveat=original['diagnostic_caveat'])
