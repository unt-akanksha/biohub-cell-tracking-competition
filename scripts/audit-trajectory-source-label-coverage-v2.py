"""Compare conservative training matches with official physical source matches."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_disagreement_data_v1 import positions


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-source-label-coverage-v2'
    assert not target.exists()
    scope=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(scope)=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    stems=[m['stem'] for m in json.loads(scope.read_text())['movies']]
    base=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    data=ROOT/'.biohub/cache/trajectory-candidate-supervision-v1'
    inv=json.loads((ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json').read_text())
    backup=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert backup['status']=='verified_backup'
    for r in backup['records']:assert sha(base/r['path'])==r['sha256']
    reference=json.loads((ROOT/'reports/experiments/trajectory-joint-source-v1-result.json').read_text())
    reference={r['stem']:r for r in reference['rows']['original']}
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1';manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files=json.loads(manifest.read_text())['files']
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import polars as pl
    import tracksdata as td
    keys=td.DEFAULT_ATTR_KEYS;target.mkdir();rows={};hashes={}
    for stem in stems:
        gp=data/(stem+'-candidates.npz');lp=data/(stem+'-labels.npz')
        assert sha(gp)==inv['candidate_sha256'][stem] and sha(lp)==inv['label_sha256'][stem]
        with np.load(gp,allow_pickle=False) as f:groups=dict(f)
        with np.load(lp,allow_pickle=False) as f:old_targets=f['target']
        selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
        for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
        final=json.loads((base/(stem+'-original')/'repaired-prediction.json').read_text())
        graph=td.graph.InMemoryGraph()
        for a in ('z','y','x'):graph.add_node_attr_key(a,pl.Float64,0.)
        graph.add_node_attr_key('source_original_id',pl.Int64,-1)
        ids=sorted(final['nodes'],key=int)
        mapped=graph.bulk_add_nodes([dict({k:final['nodes'][i][k] for k in ('t','z','y','x')},source_original_id=int(i)) for i in ids])
        mapping=dict(zip(map(int,ids),mapped))
        graph.bulk_add_edges([dict(source_id=mapping[e['source_id']],target_id=mapping[e['target_id']]) for e in final['edges']])
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_root/'train'/(stem+'.geff')))[0]
        er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        for k in scorer.COUNT_COLUMNS:assert getattr(er,k)==reference[stem][k]
        det_to_gt={int(r['source_original_id']):int(r[keys.MATCHED_NODE_ID]) for r in graph.node_attrs().iter_rows(named=True) if r[keys.MATCHED_NODE_ID] is not None and r[keys.MATCHED_NODE_ID]>=0}
        assert len(set(det_to_gt.values()))==len(det_to_gt), 'Require unique physical matches'
        gt_to_det={g:p for p,g in det_to_gt.items()}
        gt_nodes={int(r[keys.NODE_ID]):r for r in truth.node_attrs().iter_rows(named=True)}
        gt_in=defaultdict(list)
        for r in truth.edge_attrs().iter_rows(named=True):gt_in[int(r[keys.EDGE_TARGET])].append(int(r[keys.EDGE_SOURCE]))
        pp=positions(final['nodes']);tp=positions(gt_nodes)
        labels=np.full(len(groups['children']),-1,np.int64);safe=np.zeros(len(groups['parents']),bool);counts=Counter()
        for i,child in enumerate(groups['children']):
            g=det_to_gt.get(int(child))
            if g is None or len(gt_in[g])!=1:counts['no_unique_known_parent']+=1;continue
            parent=gt_in[g][0]
            if gt_nodes[g]['t']-gt_nodes[parent]['t']!=1:counts['nonconsecutive_gt']+=1;continue
            a,b=groups['offsets'][i:i+2];choices=groups['parents'][a:b]
            expected=gt_to_det.get(parent);where=np.flatnonzero(choices==expected) if expected is not None else []
            if not len(where):counts['matched_parent_outside_candidate_set']+=1;continue
            labels[i]=int(expected)
            distances=np.array([np.linalg.norm(pp[int(p)]-tp[parent]) for p in choices])
            assert distances[where[0]]<=7.+1e-6
            safe[a:b]=distances>7.;safe[a+where[0]]=True
            counts['official_positive_groups']+=1
            counts['new_positive_groups']+=int(old_targets[i]<0)
            counts['same_positive_identity']+=int(old_targets[i]==expected)
            counts['changed_positive_identity']+=int(old_targets[i]>=0 and old_targets[i]!=expected)
            counts['positive_parent_distance_above_3_25um']+=int(distances[where[0]]>3.25)
            counts['current_correct']+=int(groups['current'][i]==expected)
        counts['prior_positive_groups']=int((old_targets>=0).sum())
        path=target/(stem+'-physical-labels.npz');np.savez_compressed(path,target=labels,safe=safe)
        hashes[stem]=sha(path);rows[stem]=dict(counts)
        print(json.dumps(dict(stem=stem,**counts)),flush=True)
    result=dict(status='source_physical_label_coverage_audited',per_movie=rows,label_sha256=hashes,
                source_sha256=sha(Path(__file__)),source_scope_sha256=sha(scope),
                source_only=True,selection_or_validation_opened=False,model_fitted=False,
                node_positions_changed=False,graph_predictions_changed=False,authorized_for_submission=False,
                geometry_rule='Official unique physical distance matching; safe negatives still require >7um',
                seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-source-label-coverage-v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])))


if __name__=='__main__':main()
