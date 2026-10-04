"""Paired complete-movie diagnostic for a training-derived physical prior."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE = runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))


def score(root,notebook,truth_root,*,linker=None):
    import tracksdata as td
    import polars as pl
    import numpy as np
    from geff import GeffMetadata
    from research.independent_motion_prior import link_motion,VARIANCE,NULL_LOGIT
    originals,manifest = BASE['prepare'](root,notebook)
    candidates,receipts = {},{}
    # Freeze every complete candidate before accessing any selection labels.
    for stem,graph in originals.items():
        coords = np.asarray([[r[k] for k in ('t','z','y','x')]
            for r in graph.node_attrs().iter_rows(named=True)],dtype=float).reshape(-1,4)
        edges = (linker or link_motion)(coords)
        candidate = td.graph.InMemoryGraph()
        for axis in ('z','y','x'):
            candidate.add_node_attr_key(axis,pl.Float64,0.)
        ids = candidate.bulk_add_nodes([dict(t=int(t),z=z,y=y,x=x) for t,z,y,x in coords])
        candidate.bulk_add_edges([dict(source_id=ids[s],target_id=ids[d]) for s,d,_ in edges])
        recovered = np.asarray([[r[k] for k in ('t','z','y','x')]
            for r in candidate.node_attrs().iter_rows(named=True)],dtype=float).reshape(-1,4)
        if not np.array_equal(coords,recovered):
            raise ValueError('Motion baseline changed detections')
        candidates[stem] = candidate
        payload = json.dumps(dict(coords=coords.tolist(),edges=edges),sort_keys=True)
        receipts[stem] = dict(predicted_nodes=len(coords),predicted_edges=len(edges),
            nodes_unchanged=True,prediction_sha256=hashlib.sha256(payload.encode()).hexdigest())
    scorer = BASE['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    BASE['smoke_empty_edges'](scorer)
    rows = {'original':[],'motion':[]}
    for arm,graphs in [('original',originals),('motion',candidates)]:
        for stem,graph in graphs.items():
            path = truth_root/(stem+'.geff')
            truth = td.graph.IndexedRXGraph.from_geff(str(path))[0]
            er = scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            total = float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            recall = BASE['diagnostic_node_recall'](scorer,graph,truth)
            rows[arm].append(dict(scorer.per_sample_metrics(er,total,recall),stem=stem,embryo='6bba'))
    summaries = {arm:scorer.summarise(values) for arm,values in rows.items()}
    return dict(run_id='independent-motion-prior-v1',per_movie=rows,summaries=summaries,
        receipts=receipts,variance_um2=VARIANCE.tolist(),null_logit=NULL_LOGIT,
        by_embryo={'6bba':summaries},
        score_delta=summaries['motion']['score']-summaries['original']['score'],
        checkpoint_sha256=manifest['checkpoint_sha256'],target_audit_opened=False,
        scope='Source-embryo selection, same detections, prior fitted only on four training movies',
        authorized_for_submission=False)
