"""Freeze joint graphs, then score complete source movies; no selection truth."""
import json
import math
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,csv_equivalent_graph
from research.trajectory_joint_assignment_v1 import apply


def main():
    started=time.monotonic()
    target=ROOT/'.biohub/cache/trajectory-joint-source-v1';assert not target.exists()
    audit_path=ROOT/'reports/experiments/trajectory-candidate-ranker-v2-audit.json'
    audit=json.loads(audit_path.read_text());assert audit['qualified_for_independent_selection']==['6bba']
    training=ROOT/'.biohub/cache/trajectory-candidate-ranker-v1'
    receipt=json.loads((training/'RESULT.json').read_text())
    inventory=json.loads((ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json').read_text())
    scope=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(scope)==inventory['source_scope_sha256']=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    stems=[m['stem'] for m in json.loads(scope.read_text())['movies']]
    data=ROOT/'.biohub/cache/trajectory-candidate-supervision-v1'
    base=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    backup=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert backup['status']=='verified_backup'
    for r in backup['records']:assert sha(base/r['path'])==r['sha256']
    assert sha(training/'weights.npz')==receipt['weights_sha256']
    with np.load(training/'weights.npz',allow_pickle=False) as f:weights=f['6bba']
    target.mkdir();graphs={};details={};frozen={}
    contract=dict(selected_model='6bba',source_ranker_audit_sha256=sha(audit_path),weights_sha256=receipt['weights_sha256'],
                  assignment_sha256=sha(ROOT/'research/trajectory_joint_assignment_v1.py'),source_sha256=sha(Path(__file__)),
                  policy='One fixed degree-preserving joint assignment; protect all division/gap incident edges; no tuning',
                  source_only=True,source6_training_overlap=True,authorized_for_submission=False)
    (target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    for stem in stems:
        features=training/(stem+'-features.npz');group_path=data/(stem+'-candidates.npz')
        assert sha(features)==receipt['features_sha256'][stem] and sha(group_path)==inventory['candidate_sha256'][stem]
        original=json.loads((base/(stem+'-original')/'repaired-prediction.json').read_text())
        with np.load(features,allow_pickle=False) as f:matrix=f['features']
        with np.load(group_path,allow_pickle=False) as f:groups=dict(f)
        candidate,details[stem]=apply(original,groups,matrix,weights)
        for graph in (original,candidate):
            assert csv_equivalent_graph({int(k):v for k,v in graph['nodes'].items()},graph['edges'],100)==graph
        path=target/(stem+'-prediction.json');path.write_text(json.dumps(candidate)+'\n')
        frozen[stem]=sha(path);graphs[stem]=dict(original=original,joint=candidate)
        print(json.dumps(dict(stage='joint_graph_frozen',stem=stem,changed_edges=details[stem]['changed_edges'])),flush=True)
    (target/'PRELABEL.json').write_text(json.dumps(frozen,indent=2)+'\n')
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files=json.loads(manifest.read_text())['files']
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={a:[] for a in ('original','joint')}
    for stem in stems:
        assert sha(target/(stem+'-prediction.json'))==frozen[stem]
        selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
        for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
        truth_path=truth_root/'train'/(stem+'.geff')
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            graph=helper['prediction_graph'](graphs[stem][arm])
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            estimated=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(er,estimated,scorer.node_recall(graph,truth)),stem=stem,embryo=stem.split('_')[0])
            rows[arm].append(row)
        print(json.dumps(dict(stage='scored',stem=stem)),flush=True)
    summaries={a:scorer.summarise(r) for a,r in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in rs if r['embryo']==e]) for e in ('44b6','6bba')} for a,rs in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in rs} for a,rs in rows.items()}
    for arm in rows:
        assert summaries[arm]['n']==8 and summaries[arm]['n_adj']==8
        assert all(math.isfinite(r[k]) for r in (summaries[arm],*embryos[arm].values(),*movies[arm].values()) for k in ('score','edge_jaccard','adj_edge_jaccard'))
    gates=dict(pooled_score_strict=summaries['joint']['score']>summaries['original']['score'],
               raw_edge_strict=summaries['joint']['edge_jaccard']>summaries['original']['edge_jaccard'],
               embryo_nonregression=all(embryos['joint'][e]['score']>=embryos['original'][e]['score'] for e in embryos['joint']),
               movie_nonregression=all(movies['joint'][s]['score']>=movies['original'][s]['score'] for s in stems))
    result=dict(status='complete_source_graph_diagnostic',rows=rows,summaries=summaries,by_embryo=embryos,
                per_movie=movies,changes=details,graph_sha256=frozen,contract_sha256=sha(target/'CONTRACT.json'),
                gates=gates,passed_for_independent_selection=all(gates.values()),source6_training_overlap=True,
                source_only=True,selection_or_validation_read=False,authorized_for_submission=False,
                seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    (ROOT/'reports/experiments/trajectory-joint-source-v1-result.json').write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(helper['finite'](dict(status=result['status'],summaries=summaries,gates=gates,seconds=result['seconds']))))


if __name__=='__main__':main()
