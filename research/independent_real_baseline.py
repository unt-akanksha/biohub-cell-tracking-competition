"""Split and training-loop safeguards for a fresh real-domain baseline."""
import hashlib


DIAGNOSTIC_STEMS = {'44b6_24264f12', '44b6_81c256f0', '6bba_23af9eeb', '6bba_f1fde7e0'}


def stable_key(stem):
    return hashlib.sha256(('biohub-independent-real-v1:' + stem).encode()).hexdigest()


def make_split(stems):
    stems = list(stems)
    if len(set(stems)) != len(stems):
        raise ValueError('Duplicate movie identity')
    if any(len(s.split('_')) != 2 or s.split('_')[0] not in {'44b6', '6bba'} for s in stems):
        raise ValueError('Unexpected embryo or movie identity')
    groups = {e: sorted((s for s in stems if s.startswith(e + '_')), key=stable_key)
              for e in ('44b6', '6bba')}
    if any(len(v) < 12 for v in groups.values()):
        raise ValueError('Insufficient source selection and target audit movies')
    folds = []
    for fold, target in enumerate(groups):
        source = next(e for e in groups if e != target)
        # Selection is from the training embryo, never the held-out embryo.
        selection = groups[source][:8]
        train = groups[source][8:]
        audit = [s for s in groups[target] if s not in DIAGNOSTIC_STEMS]
        folds.append(dict(fold=fold, training_embryo=source, held_out_embryo=target,
                          train=train, selection=selection, audit_order=audit,
                          initial_complete_movie_audit=audit[:4]))
    return dict(run_id='independent-real-baseline-v1', folds=folds,
                initialization='random; no Biohub/public checkpoint warm start',
                role_assignment='SHA-256 movie-name ordering; no labels or scores',
                audit_is_untouched_by_all_prior_research=False,
                audit_scope='Embryo excluded from this fold training and checkpoint selection; prior project exposure is not erased',
                production_promotion_authorized=False)


def fresh_batches(loader):
    """Repeat loader passes without itertools.cycle retaining image batches."""
    while True:
        seen = False
        for batch in loader:
            seen = True
            yield batch
        if not seen:
            raise ValueError('Empty training loader')


def patch_train_epoch(source):
    """Patch only the pinned organizer epoch function, failing on source drift.

    The caller supplies GradScaler and fresh_batches in its namespace, and an
    AMP encoder that returns FP32 features/logits for matching and sparse loss.
    """
    replacements = {
        '_cycle(loader)': 'fresh_batches(loader)',
        'loss.backward()': 'scaler.scale(loss).backward()',
        'torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)':
            'scaler.unscale_(optimizer)\n        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)',
        'optimizer.step()': 'scaler.step(optimizer)\n        scaler.update()',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('Organizer train_epoch source drift: ' + old)
        source = source.replace(old, new)
    compile(source, '<independent-real-train-epoch>', 'exec')
    return source


def install_empty_attention_guard(model):
    """Skip undefined attention on empty key sets; preserve nonempty samples.

    Empty pairs carry no association supervision. No detection or graph node
    is introduced, removed, or changed, and all real-key attention is unchanged.
    """
    import torch
    for block in model.transformer.blocks:
        if getattr(block, '_biohub_empty_guard', False):
            continue
        original = block.forward
        def guarded(q, kv, kv_mask=None, _original=original):
            if kv_mask is None:
                return _original(q, kv, kv_mask)
            active = kv_mask.any(dim=1)
            if bool(active.all()):
                return _original(q, kv, kv_mask)
            if not bool(active.any()):
                return q
            indices = torch.nonzero(active, as_tuple=False).flatten()
            valid = _original(q[indices], kv[indices], kv_mask[indices])
            return q.index_copy(0, indices, valid)
        block.forward = guarded
        block._biohub_empty_guard = True
