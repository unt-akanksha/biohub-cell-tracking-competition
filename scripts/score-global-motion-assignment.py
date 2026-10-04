"""Frozen D4 nodes: existing static greedy vs fixed ordinary-link LAP, CPU only."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from global_motion_assignment import contract,link
from independent_motion_prior import link_motion
BASE_SHA='db9d75ad43a9bc74d3f38f3aef48e5e617510a6abb93205a0387e726415d080f'


def main():
    import numpy as np
    import polars as pl
    import tracksdata as td
    from geff import GeffMetadata
    from research.empty_graph_schema import restore_empty_spatial_schema
    score_module=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    scorer=score_module['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    tree_hash=runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))['tree_hash']
    report_path=ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json'
    if hashlib.sha256(report_path.read_bytes()).hexdigest()!=BASE_SHA: raise ValueError('Frozen baseline changed')
    frozen=json.loads(report_path.read_text())['result']
    cache=ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-selection-v1/detector_spatial_tta_selection/outputs'
    manifest=json.loads((cache/'selection_manifest.json').read_text())
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    if [r['stem'] for r in manifest['records']]!=split['folds'][0]['selection']: raise ValueError('Scope changed')
    started=time.monotonic(); records={'static_greedy':[],'static_lap':[]}
    for source_record in manifest['records']:
        stem=source_record['stem']; path=cache/(stem+'.geff')
        if tree_hash(path)!=source_record['graph_sha256']: raise ValueError('Frozen graph changed')
        original=td.graph.IndexedRXGraph.from_geff(str(path))[0]
        coords=original.node_attrs().select('t','z','y','x').to_numpy().astype(float)
        # Both fixed alternatives are generated before this movie's truth is read.
        proposals={'static_greedy':[(s,d) for s,d,p in link_motion(coords)],'static_lap':link(coords)}
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
        estimated=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
        for arm,edges in proposals.items():
            graph=td.graph.InMemoryGraph()
            for k in ('z','y','x'): graph.add_node_attr_key(k,pl.Float64,-999999.)
            ids=graph.bulk_add_nodes([dict(t=int(t),z=float(z),y=float(y),x=float(x)) for t,z,y,x in coords])
            if list(ids)!=original.node_attrs()['node_id'].to_list(): raise ValueError('Node IDs changed')
            if edges: graph.bulk_add_edges([dict(source_id=ids[s],target_id=ids[d]) for s,d in edges])
            restore_empty_spatial_schema(graph)
            if not np.array_equal(graph.node_attrs().select('t','z','y','x').to_numpy(),coords): raise ValueError('Node coordinates changed')
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            row=dict(scorer.per_sample_metrics(er,estimated,score_module['diagnostic_node_recall'](scorer,graph,truth)),stem=stem,embryo='6bba')
            records[arm].append(row)
            print(json.dumps(dict(arm=arm,**row)),flush=True)
    summaries={k:scorer.summarise(rows) for k,rows in records.items()}
    deltas={label:{key:summaries['static_lap'][key]-ref[key] for key in ('score','edge_jaccard','node_recall','division_jaccard')}
        for label,ref in [('static_greedy',summaries['static_greedy']),('frozen_image_flow',frozen['summary'])]}
    paired={label:[dict(stem=r['stem'],score_delta=r['adj_edge_jaccard']-b['adj_edge_jaccard']) for r,b in zip(records['static_lap'],base)]
        for label,base in [('static_greedy',records['static_greedy']),('frozen_image_flow',frozen['per_movie'])]}
    result=dict(status='completed_static_assignment_source_ablation',contract=contract(),summaries=summaries,
        per_movie=records,deltas=deltas,paired=paired,baseline_report_sha256=BASE_SHA,elapsed_seconds=time.monotonic()-started,
        candidate_source_sha256=hashlib.sha256((ROOT/'research/global_motion_assignment.py').read_bytes()).hexdigest(),
        nodes_preserved=True,target_audit_opened=False,authorized_for_submission=False,
        limitation='No divisions can be predicted by this one-to-one ablation. Not the full Jaqaman algorithm, not an embryo audit or submission candidate.')
    (ROOT/'reports/experiments/global-motion-assignment-v1-score.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],summaries=summaries,deltas=deltas,elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()
