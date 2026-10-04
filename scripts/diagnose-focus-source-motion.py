"""Re-score and attribute the frozen source models; no prediction changes."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_motion_error_attribution import classify_unlinked
SOURCE=runpy.run_path(str(ROOT/'scripts/score-focus-source-flow.py'))
DECOMPOSE=runpy.run_path(str(ROOT/'scripts/diagnose-frozen-detector-selection.py'))['decompose_edges']
REF_SHA='896d3de86fa30fcd860f2c16073d47718118d94e1a43fe70710626c7b32ea169'
NB_SHA='d36fd6ef5fb460ae4964429aded2832b0d44db91a65836476ecea64639f30375'


def main():
    import numpy as np
    import tracksdata as td
    target=ROOT/'reports/experiments/focus-source-motion-error-attribution.json'
    if target.exists(): raise ValueError('Do not overwrite completed diagnostic')
    path=ROOT/'reports/experiments/focus-source-flow-v1-result.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=REF_SHA: raise ValueError('Frozen reference changed')
    reference=json.loads(path.read_text())
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-source-flow-v1/focus_source_flow'
    prepared,parents,policy=SOURCE['prepare'](folder,ROOT/'kaggle/biohub-focus-source-flow-v1/biohub-focus-source-flow-v1.ipynb',NB_SHA)
    metric=SOURCE['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    rows=[]
    for index,stem in enumerate(policy['source_stems']):
        for arm in ('parent','control','candidate'):
            graph=parents[stem] if arm=='parent' else td.graph.IndexedRXGraph.from_geff(str(prepared[stem][arm]))[0]
            truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
            result=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            expected=reference['per_movie'][arm][index]
            if any(getattr(result,k)!=expected[k] for k in result._fields): raise ValueError('Original official counts did not replay')
            keys=td.DEFAULT_ATTR_KEYS
            nodes=graph.node_attrs().sort(keys.NODE_ID)
            mapping=dict(nodes.select(keys.NODE_ID,keys.MATCHED_NODE_ID).iter_rows())
            gt_edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
            edges=list(graph.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
            counts=DECOMPOSE(gt_edges,mapping,edges)
            if counts['recovered']!=result.edge_tp or counts['total_gt_edges']-counts['recovered']!=result.edge_fn:
                raise ValueError('Diagnostic partition differs from official counts')
            row=dict(stem=stem,arm=arm,embryo=stem.split('_')[0],**counts,official_edge_fp=result.edge_fp)
            if arm=='candidate':
                with np.load(folder/'outputs'/stem/'sampled_flow.npz',allow_pickle=False) as data:
                    coords,flow=data['coords'].copy(),data['backward_um'].copy()
                if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords): raise ValueError('Raw coordinate ordering changed')
                index_by_id={int(v):i for i,v in enumerate(nodes[keys.NODE_ID])}
                indexed_mapping={index_by_id[k]:v for k,v in mapping.items()}
                diagnostic=classify_unlinked(coords,flow,indexed_mapping,[(index_by_id[s],index_by_id[d]) for s,d in edges],gt_edges)
                if diagnostic['both_detected_unlinked']!=counts['both_endpoints_detected_but_not_linked']:
                    raise ValueError('Conditional association diagnosis is not exhaustive')
                row['motion_failure_partition']=diagnostic
            rows.append(row); print(json.dumps(row),flush=True)
    totals={}
    for arm in ('parent','control','candidate'):
        selected=[r for r in rows if r['arm']==arm]
        totals[arm]={k:sum(r[k] for r in selected) for k,v in selected[0].items() if isinstance(v,int)}
    motion={k:sum(r['motion_failure_partition'][k] for r in rows if r['arm']=='candidate')
            for k in next(r['motion_failure_partition'] for r in rows if r['arm']=='candidate')}
    result=dict(status='verified_source_motion_error_attribution',reference_sha256=REF_SHA,per_movie=rows,
        totals=totals,motion_failure_totals=motion,predictions_modified=False,new_target_movies_opened=0,
        gpu_seconds=0,authorized_for_submission=False,
        caveat='Already-exposed source development data; endpoint availability is not proof an algorithm can recover a link. Unmatched predictions are not automatically false detections under sparse annotations.')
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(totals=totals,motion_failure_totals=motion)),flush=True)


if __name__=='__main__': main()
