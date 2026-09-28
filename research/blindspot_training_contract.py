"""Frozen training-only scope and proxy gate; not a tracking promotion gate."""
import hashlib
import json
import math

SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
SEED = 20260910
PATCH = (32,64,64)
FRAMES = (0,49,99)


def identity(split_bytes):
    if hashlib.sha256(split_bytes).hexdigest() != SPLIT_SHA:
        raise ValueError('Frozen original split required')
    fold = json.loads(split_bytes)['folds'][0]
    diagnostic = fold['train'][::5]
    fitting = [s for s in fold['train'] if s not in diagnostic]
    if (len(set(fitting)) != 96 or len(set(diagnostic)) != 24
            or set(fitting)&set(diagnostic) or not all(s.startswith('6bba_') for s in fold['train'])
            or set(fold['train'])&set(fold['selection']+fold['audit_order'])):
        raise ValueError('Training-only96/24 split required')
    return dict(run_id='blindspot-real-probe-v1',fitting_stems=fitting,diagnostic_stems=diagnostic,
        split_sha256=SPLIT_SHA,seed=SEED,frames=list(FRAMES),patch_shape=list(PATCH),steps=100,
        batch_size=4,width=16,normalization='Fixed affine fitting-pixel quantiles0.001/0.999; no clipping',
        optimizer=dict(name='Adam',lr=.001),mixed_precision=True,loss='Interior7-voxel-cropped MSE to noisy input',
        public_checkpoint_loaded=False,ground_truth_opened=False,selection_opened=False,
        target_embryo_images_read=False,authorized_for_submission=False)


def origin(stem,frame):
    import numpy as np
    if frame not in FRAMES:
        raise ValueError('Undeclared image frame')
    seed = int.from_bytes(hashlib.sha256(f'{SEED}:{stem}:{frame}'.encode()).digest()[:8],'little')
    rng = np.random.default_rng(seed)
    return tuple(int(rng.integers(0,n-p+1)) for n,p in zip((64,256,256),PATCH))


def proxy_gate(rows,diagnostic_stems):
    if len(diagnostic_stems) != 24 or [r['stem'] for r in rows] != diagnostic_stems:
        raise ValueError('Exact24 diagnostic movies required')
    for row in rows:
        if (row['pixels'] != 3*18*50*50 or any(not math.isfinite(row[k]) or row[k]<0
                                            for k in ('model_sse','neighbor_sse'))):
            raise ValueError('Complete finite diagnostic errors required')
    count = sum(r['pixels'] for r in rows)
    model = sum(r['model_sse'] for r in rows)/count
    control = sum(r['neighbor_sse'] for r in rows)/count
    gains = sum(r['model_sse'] < r['neighbor_sse'] for r in rows)
    return dict(model_mse=model,neighbor_mse=control,improved_movies=gains,
                proxy_gate_passed=model < control and gains >= 18,
                authorized_for_submission=False,
                caveat='Self-supervised proxy only; actual noise independence and tracking gain unverified.')
