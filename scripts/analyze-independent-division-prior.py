"""Training-only diagnosis of fixed motion/null bias; no threshold fitting.

Evaluate true annotated displacements, not model predictions. The null-only
posterior is an optimistic upper bound on the pure-prior parent probability:
additional competing parents can only reduce it. A learned residual may offset
this prior, so these counts are not predicted misses or validation metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from independent_motion_prior import SCALE, VARIANCE, NULL_LOGIT


def summarize(displacements):
    values = np.asarray(displacements, dtype=float).reshape(-1, 3)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty displacements required')
    logits = -.5 * (values**2 / VARIANCE).sum(1)
    residual = np.maximum(NULL_LOGIT - logits, 0.)
    probabilities = np.exp(-np.logaddexp(0., NULL_LOGIT - logits))
    quantiles = lambda a: dict(zip(('p50','p90','p95','p99'), np.quantile(a, [.5,.9,.95,.99]).tolist()))
    return dict(edges=len(values), prior_not_above_null=int((logits <= NULL_LOGIT).sum()),
        fraction_prior_not_above_null=float((logits <= NULL_LOGIT).mean()),
        displacement_um=quantiles(np.linalg.norm(values,axis=1)),
        residual_required_to_exceed_null=quantiles(residual),
        null_only_parent_posterior_quantiles=quantiles(probabilities))


def run():
    import tracksdata as td
    split_path = ROOT / 'research/independent_real_baseline_v1_split.json'
    split = json.loads(split_path.read_text())
    fold = split['folds'][0]
    stems = fold['train']
    if len(stems) != 120 or set(stems) & set(fold['selection'] + fold['audit_order']):
        raise ValueError('Frozen training-only scope required')
    pools = {'single_child': [], 'division_daughter': []}
    rows = []
    for stem in stems:
        if not stem.startswith(fold['training_embryo'] + '_'):
            raise ValueError('Wrong training embryo')
        path = ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train' / (stem + '.geff')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        nodes = {int(row['node_id']):row for row in graph.node_attrs().iter_rows(named=True)}
        children = {}
        for edge in graph.edge_attrs().iter_rows(named=True):
            src, dst = int(edge['source_id']), int(edge['target_id'])
            if nodes[dst]['t'] == nodes[src]['t'] + 1:
                children.setdefault(src, []).append(dst)
        division_count = blocked_divisions = 0
        for parent, daughters in children.items():
            if len(daughters) not in (1, 2):
                raise ValueError('Unexpected annotated child degree')
            delta = np.asarray([[nodes[d][axis] - nodes[parent][axis] for axis in ('z','y','x')]
                                for d in daughters], dtype=float) * SCALE
            key = 'single_child' if len(daughters) == 1 else 'division_daughter'
            pools[key].extend(delta.tolist())
            if len(daughters) == 2:
                division_count += 1
                blocked_divisions += int(np.any(-.5*(delta**2/VARIANCE).sum(1) <= NULL_LOGIT))
        rows.append(dict(stem=stem, division_events=division_count,
                         divisions_with_any_daughter_prior_not_above_null=blocked_divisions))
    return dict(scope='Frozen 120 training movies; true displacements only, not predicted detections',
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        prior=dict(variance_um2=VARIANCE.tolist(),null_logit=NULL_LOGIT,scale=SCALE.tolist()),
        summaries={key:summarize(values) for key,values in pools.items()},per_movie=rows,
        division_events=sum(r['division_events'] for r in rows),
        divisions_with_any_daughter_prior_not_above_null=sum(r['divisions_with_any_daughter_prior_not_above_null'] for r in rows),
        caveat='Learned residuals may overcome this prior; no actual model error rate is inferred',
        selection_opened=False,target_audit_opened=False,model_changed=False,authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result,indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('summaries','division_events','divisions_with_any_daughter_prior_not_above_null')},indent=2))
