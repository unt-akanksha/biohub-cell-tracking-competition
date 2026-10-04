"""CPU-only paired assignment experiment on immutable owned flow caches."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from learned_global_motion_assignment import contract,link
from research.backward_flow_linking import link_backward_flow
FLOW_MANIFEST_SHA='c6d1811ee17e25c0baaba64675883942a6c2fd13e5cb0b7c0f84c79887b7f744'


def main():
    import numpy as np
    import polars as pl
    import tracksdata as td
    from geff import GeffMetadata
    from research.empty_graph_schema import restore_empty_spatial_schema
    module=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    scorer=module['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    root=ROOT/'.biohub/cache/kernel-outputs/backward-flow-selection-v1/backward_flow_selection/outputs'
    path=root/'flow_manifest.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FLOW_MANIFEST_SHA: raise ValueError('Frozen flow manifest changed')
    manifest=json.loads(path.read_text())
    reference=json.loads((ROOT/'reports/experiments/backward-flow-selection-v1-score.json').read_text())['result']
    strongest=json.loads((ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json').read_text())['result']
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    if ([r['stem'] for r in manifest['records']]!=split['folds'][0]['selection']
        or manifest['checkpoint_sha256']!=contract()['flow_checkpoint_sha256']): raise ValueError('Source-only flow scope changed')
    records={'flow_greedy':[],'flow_lap':[]}; started=time.monotonic()
    for rec,expected in zip(manifest['records'],reference['per_movie']['motion']):
        stem=rec['stem']; path=root/(stem+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['file_sha256']: raise ValueError('Motion cache changed')
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!={'coords','backward_um'}: raise ValueError('Unexpected cached fields')
            coords,flow=saved['coords'].copy(),saved['backward_um'].copy()
        if (coords.shape!=(rec['predicted_nodes'],4) or rec['processed_frames']!=100
            or rec['processed_pairs']!=99 or np.any(coords>=rec['image_shape']) or stem!=expected['stem']):
            raise ValueError('Incomplete or mismatched cached movie')
        alternatives={'flow_greedy':[(s,d) for s,d,p in link_backward_flow(coords,flow)],'flow_lap':link(coords,flow)}
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
        total=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
        for arm,edges in alternatives.items():
            graph=td.graph.InMemoryGraph()
            for k in ('z','y','x'): graph.add_node_attr_key(k,pl.Float64,-999999.)
            ids=graph.bulk_add_nodes([dict(t=int(t),z=float(z),y=float(y),x=float(x)) for t,z,y,x in coords])
            if edges: graph.bulk_add_edges([dict(source_id=ids[s],target_id=ids[d]) for s,d in edges])
            restore_empty_spatial_schema(graph)
            if not np.array_equal(coords,graph.node_attrs().select('t','z','y','x').to_numpy()): raise ValueError('Cached detections changed')
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            if arm=='flow_greedy' and any(getattr(er,k)!=expected[k] for k in er._fields):
                raise ValueError('Original flow control did not replay exact official counts')
            row=dict(scorer.per_sample_metrics(er,total,module['diagnostic_node_recall'](scorer,graph,truth)),stem=stem,embryo='6bba')
            records[arm].append(row); print(json.dumps(dict(arm=arm,**row)),flush=True)
    summaries={arm:scorer.summarise(rows) for arm,rows in records.items()}
    deltas={label:{k:summaries['flow_lap'][k]-ref[k] for k in ('score','edge_jaccard','node_recall','division_jaccard')}
        for label,ref in [('paired_flow_greedy',summaries['flow_greedy']),('strongest_detector_d4',strongest['summary'])]}
    result=dict(status='completed_learned_assignment_source_ablation',contract=contract(),per_movie=records,
        summaries=summaries,deltas=deltas,control_counts_exactly_replayed=True,flow_manifest_sha256=FLOW_MANIFEST_SHA,
        regressions_vs_paired=[r['stem'] for r,b in zip(records['flow_lap'],records['flow_greedy']) if r['adj_edge_jaccard']<b['adj_edge_jaccard']-1e-12],
        worst_movie=min(records['flow_lap'],key=lambda r:r['adj_edge_jaccard']),elapsed_seconds=time.monotonic()-started,
        source_sha256={name:hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest() for name in ('learned_global_motion_assignment.py','global_motion_assignment.py')},
        authorized_for_submission=False,target_audit_opened=False,
        caveat='Fixed native c502 nodes, not D4 nodes. Comparison with paired flow isolates solver; comparison with D4 is cross-candidate. No division modeling in LAP; source development only.')
    (ROOT/'reports/experiments/learned-global-motion-assignment-v1-score.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','summaries','deltas','regressions_vs_paired','elapsed_seconds')}),flush=True)


if __name__=='__main__': main()
