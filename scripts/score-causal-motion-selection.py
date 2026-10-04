"""Complete cached-node test of a frozen training-only causal predictor."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def validate_fit(receipt, split_bytes):
    fold = json.loads(split_bytes)['folds'][0]
    diagnostic = set(fold['train'][::5])
    if (receipt['split_sha256'] != hashlib.sha256(split_bytes).hexdigest()
            or receipt['fitting_stems'] != [s for s in fold['train'] if s not in diagnostic]
            or receipt['diagnostic_stems'] != [s for s in fold['train'] if s in diagnostic]
            or any(receipt[k] is not False for k in ('selection_opened', 'target_audit_opened', 'model_changed', 'authorized_for_submission'))):
        raise ValueError('Exact training-only frozen fit required')
    return receipt['model']


def score(root, notebook, truth_root):
    from research.causal_motion_prior import link_causal_motion
    from research.independent_motion_prior import VARIANCE
    path = ROOT/'reports/experiments/independent-motion-persistence-training.json'
    receipt = json.loads(path.read_text())
    model = validate_fit(receipt, (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    manifest = json.loads((root/'outputs/selection_manifest.json').read_text())
    if (manifest['checkpoint_sha256'] != CHECKPOINT
            or manifest['manifest_sha256'] != receipt['split_sha256']):
        raise ValueError('Wrong native joint checkpoint or split')
    result = runpy.run_path(str(ROOT/'scripts/score-independent-motion-prior.py'))['score'](
        root, notebook, truth_root, linker=lambda points: link_causal_motion(points, model))
    result.update(run_id='causal-motion-selection-v1', model=model,
                  fit_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope='Full source-selection movies; causal predicted history; fit96 training movies only',
                  fallback_variance_um2=VARIANCE.tolist())
    result.pop('variance_um2')
    result['adjusted_edge_regressions'] = [c['stem'] for c, p in zip(
        result['per_movie']['motion'], result['per_movie']['original'])
        if c['adj_edge_jaccard'] < p['adj_edge_jaccard'] - 1e-12]
    return result
