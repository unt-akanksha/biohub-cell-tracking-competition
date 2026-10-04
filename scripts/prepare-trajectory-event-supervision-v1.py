"""Build source-only sparse physical labels after prediction/features are frozen."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_supervision_v1 import label


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);args=p.parse_args()
    started=time.monotonic();name='trajectory-event-source-v1-b'+str(args.batch)
    feature_root=ROOT/'.biohub/cache'/(name+'-features')
    feature_report=json.loads((feature_root/'RESULT.json').read_text())
    assert feature_report['status']=='current_candidate_source_features_frozen' and not feature_report['ground_truth_opened']
    scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan'/('batch-'+str(args.batch))/'MOVIES.json'
    assert sha(scope)==feature_report['source_scope_sha256']
    plan=json.loads(scope.read_text());stems=[m['stem'] for m in plan['movies']]
    assert all(m['role']=='optimization' for m in plan['movies']) and set(stems)==set(feature_report['per_movie'])
    prior=json.loads((ROOT/'reports/experiments'/(name+'-capacity.json')).read_text())
    assert prior['source_scope_sha256']==sha(scope) and prior['source_only']
    target=ROOT/'.biohub/cache'/(name+'-supervision');assert not target.exists()
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files=json.loads(manifest.read_text())['files']
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import polars as pl
    import tracksdata as td
    keys=td.DEFAULT_ATTR_KEYS;records={};pooled=Counter();target.mkdir()
    for stem in stems:
        selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
        for row in selected:assert sha(truth_root/'train'/row['relative_path'])==row['sha256']
        frozen=feature_report['per_movie'][stem]
        prediction_path=feature_root/(stem+'-prediction.json');groups_path=feature_root/(stem+'-groups.npz')
        assert sha(prediction_path)==frozen['prediction_sha256'] and sha(groups_path)==frozen['groups_sha256']
        final=json.loads(prediction_path.read_text())
        with np.load(groups_path,allow_pickle=False) as data:groups=dict(data)
        graph=td.graph.InMemoryGraph()
        for axis in ('z','y','x'):graph.add_node_attr_key(axis,pl.Float64,0.)
        graph.add_node_attr_key('source_original_id',pl.Int64,-1)
        ids=sorted(final['nodes'],key=int)
        mapped=graph.bulk_add_nodes([dict({k:final['nodes'][i][k] for k in ('t','z','y','x')},source_original_id=int(i)) for i in ids])
        mapping=dict(zip(map(int,ids),mapped))
        graph.bulk_add_edges([dict(source_id=mapping[e['source_id']],target_id=mapping[e['target_id']]) for e in final['edges']])
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_root/'train'/(stem+'.geff')))[0]
        result=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        for key in ('division_tp','division_fp','division_fn'):assert getattr(result,key)==prior['per_movie'][stem][key.replace('division_','official_division_')]
        pairs=[(int(r['source_original_id']),int(r[keys.MATCHED_NODE_ID])) for r in graph.node_attrs().iter_rows(named=True)
               if r[keys.MATCHED_NODE_ID] is not None and r[keys.MATCHED_NODE_ID]>=0]
        assert len({g for _,g in pairs})==len(pairs)==len({d for d,_ in pairs})
        truth_nodes={int(r[keys.NODE_ID]):r for r in truth.node_attrs().iter_rows(named=True)}
        truth_edges=[(int(r[keys.EDGE_SOURCE]),int(r[keys.EDGE_TARGET])) for r in truth.edge_attrs().iter_rows(named=True)]
        targets,safe,counts=label(groups,final['nodes'],truth_nodes,truth_edges,dict(pairs))
        outgoing=defaultdict(list)
        for a,b in truth_edges:
            if truth_nodes[b]['t']==truth_nodes[a]['t']+1:outgoing[a].append(b)
        counts['annotated_consecutive_parent_opportunities']=sum(len(v) in (1,2) for v in outgoing.values())
        counts['annotated_division_parents']=sum(len(v)==2 for v in outgoing.values())
        path=target/(stem+'-labels.npz')
        np.savez_compressed(path,target=targets,safe=safe)
        records[stem]=dict(counts=counts,labels_sha256=sha(path),prediction_sha256=sha(prediction_path),groups_sha256=sha(groups_path))
        pooled.update(counts)
        print(json.dumps(dict(stem=stem,**counts)),flush=True)
    report=dict(status='source_event_supervision_prepared',per_movie=records,pooled=dict(pooled),
                source_scope_sha256=sha(scope),feature_receipt_sha256=sha(feature_root/'RESULT.json'),
                source_sha256=sha(Path(__file__)),label_helper_sha256=sha(ROOT/'research/trajectory_event_supervision_v1.py'),
                source_only=True,ground_truth_files_opened=True,selection_or_validation_opened=False,
                missing_parents_are_unknown_not_births=True,unannotated_children_are_not_negative=True,
                predictions_unchanged=True,model_fitted=False,authorized_for_submission=False,
                seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (ROOT/'reports/experiments'/(name+'-supervision.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],pooled=report['pooled'],seconds=report['seconds'])))


if __name__=='__main__':main()
