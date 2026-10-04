"""Training-only movie-held-out test of motion persistence, without images.

Uses true consecutive nondivision triplets to test a causal motion hypothesis.
This is an optimistic GT-coordinate diagnostic, not a tracking score. Fitting
and diagnostic movies are separate subsets of the original120 training movies;
the eight source-selection movies and target-embryo audit remain unopened.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.asarray([1.625,.40625,.40625])


def fit(previous, following):
    previous, following = np.asarray(previous,float), np.asarray(following,float)
    if previous.shape != following.shape or previous.ndim != 2 or previous.shape[1] != 3 or len(previous) < 2:
        raise ValueError('Paired 3D training velocities required')
    if not np.isfinite(previous).all() or not np.isfinite(following).all():
        raise ValueError('Finite velocities required')
    # Conservative axis-wise shrinkage: no velocity amplification or reversal.
    pmean, fmean = previous.mean(0), following.mean(0)
    covariance = ((previous-pmean)*(following-fmean)).mean(0)
    variance = ((previous-pmean)**2).mean(0)
    alpha = np.clip(covariance/np.maximum(variance,1e-12),0.,1.)
    intercept = fmean-alpha*pmean
    residual = following-(previous*alpha+intercept)
    return dict(alpha=alpha.tolist(),intercept_um=intercept.tolist(),
                residual_variance_um2=residual.var(0).tolist(),fit_triplets=len(previous))


def error(prediction, target):
    residual = np.asarray(prediction)-np.asarray(target)
    length = np.linalg.norm(residual,axis=1)
    return dict(n=len(length),mae_per_coordinate_um=float(np.abs(residual).mean()),
                rmse_per_coordinate_um=float(np.sqrt((residual**2).mean())),
                median_endpoint_um=float(np.median(length)),p90_endpoint_um=float(np.quantile(length,.9)))


def run():
    import tracksdata as td
    split_path = ROOT/'research/independent_real_baseline_v1_split.json'
    fold = json.loads(split_path.read_text())['folds'][0]
    train = fold['train']
    if len(train) != 120 or set(train)&set(fold['selection']+fold['audit_order']):
        raise ValueError('Frozen training-only scope required')
    # Use the already frozen movie order, without inspecting labels or scores.
    diagnostic = set(train[::5])
    groups = dict(fit=[],diagnostic=[])
    per_movie = {}
    for stem in train:
        path = ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        nodes = {int(r['node_id']):r for r in graph.node_attrs().iter_rows(named=True)}
        incoming, outgoing = {},{}
        for edge in graph.edge_attrs().iter_rows(named=True):
            source,target = int(edge['source_id']),int(edge['target_id'])
            if nodes[target]['t'] == nodes[source]['t']+1:
                incoming.setdefault(target,[]).append(source)
                outgoing.setdefault(source,[]).append(target)
        samples = []
        for middle, parents in incoming.items():
            children = outgoing.get(middle,[])
            if len(parents) != 1 or len(children) != 1 or len(outgoing[parents[0]]) != 1:
                continue
            previous = [(nodes[middle][axis]-nodes[parents[0]][axis])*s for axis,s in zip(('z','y','x'),SCALE)]
            following = [(nodes[children[0]][axis]-nodes[middle][axis])*s for axis,s in zip(('z','y','x'),SCALE)]
            samples.append((previous,following))
        if not samples:
            raise ValueError('No causal nondivision triplets in '+stem)
        per_movie[stem] = samples
        groups['diagnostic' if stem in diagnostic else 'fit'].extend(samples)
    x,y = np.asarray(groups['fit']).transpose(1,0,2)
    model = fit(x,y)
    alpha,intercept = np.asarray(model['alpha']),np.asarray(model['intercept_um'])
    rows = []
    for stem in train:
        if stem not in diagnostic:
            continue
        px,py = np.asarray(per_movie[stem]).transpose(1,0,2)
        rows.append(dict(stem=stem,zero=error(np.zeros_like(py),py),
                         velocity=error(px,py),shrinkage=error(px*alpha+intercept,py)))
    dx,dy = np.asarray(groups['diagnostic']).transpose(1,0,2)
    return dict(scope='96 fitting and24 diagnostic movies inside original120 training movies',
        fitting_stems=[s for s in train if s not in diagnostic],diagnostic_stems=[s for s in train if s in diagnostic],
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),model=model,
        diagnostic=dict(zero=error(np.zeros_like(dy),dy),velocity=error(dx,dy),shrinkage=error(dx*alpha+intercept,dy)),
        per_movie=rows,selection_opened=False,target_audit_opened=False,model_changed=False,
        caveat='True past links and GT coordinates; detection noise and wrong past links are absent',
        authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(model=result['model'],diagnostic=result['diagnostic']),indent=2))
