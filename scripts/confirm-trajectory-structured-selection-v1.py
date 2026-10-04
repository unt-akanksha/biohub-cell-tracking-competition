"""One fixed independent diagnostic; preserve source failure, no automatic promotion."""
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
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-structured-selection-v1'
    assert not target.exists()
    feature_root=ROOT/'.biohub/cache/trajectory-selection-features-v1'
    proof=json.loads((feature_root/'RESULT.json').read_text())
    assert proof['status']=='selection_features_frozen_no_labels' and not proof['ground_truth_opened']
    scope=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-plan/MOVIES.json'
    assert sha(scope)==proof['scope_sha256']=='1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e'
    plan=json.loads(scope.read_text());stems=[m['stem'] for m in plan['movies']]
    assert len(stems)==10 and all(m['role']=='selection' for m in plan['movies'])
    base=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-full-output'
    backup=json.loads((ROOT/'reports/experiments/trajectory-ranker-selection-v1-full-harvest.json').read_text())
    assert backup['status']=='verified_backup'
    for r in backup['records']:assert sha(base/r['path'])==r['sha256']
    assert sha(base/'result.json')==proof['terminal_sha256']
    model=ROOT/'.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
    model_report=json.loads((ROOT/'reports/experiments/trajectory-structured-loss-v1-full.json').read_text())
    assert sha(model)==model_report['weights_sha256']=='e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
    source_report=ROOT/'reports/experiments/trajectory-structured-joint-source-v1-result.json'
    source=json.loads(source_report.read_text())
    assert source['gates']==dict(pooled_score_strict=True,raw_edge_strict=True,embryo_nonregression=True,movie_nonregression=False)
    assert sha(ROOT/'research/trajectory_joint_assignment_v1.py')=='4d1cf75a28b7a051fddfd738430be8af8e41bc7c342cdcabfcc53a96dcc81b99'
    with np.load(model,allow_pickle=False) as f:weights=f['6bba']
    target.mkdir();contract=dict(model_sha256=sha(model),source_model='6bba',source_report_sha256=sha(source_report),
        source_movie_gate_failed=True,source_failure_not_reclassified=True,source_scope_sha256=sha(scope),
        reason='One fixed independent diagnostic of strongest actual pooled source gain; not source gate relaxation or model tuning',
        source_sha256=sha(Path(__file__)),authorized_for_submission=False,automatic_promotion_forbidden=True)
    (target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    graphs={};frozen={};changes={}
    for stem in stems:
        record=proof['per_movie'][stem];gp=feature_root/(stem+'-groups.npz');fp=feature_root/(stem+'-features.npz')
        bp=base/(stem+'-original')/'repaired-prediction.json'
        assert sha(gp)==record['groups_sha256'] and sha(fp)==record['features_sha256'] and sha(bp)==record['baseline_sha256']
        original=json.loads(bp.read_text())
        with np.load(gp,allow_pickle=False) as f:groups=dict(f)
        with np.load(fp,allow_pickle=False) as f:matrix=f['features']
        candidate,changes[stem]=apply(original,groups,matrix,weights)
        assert csv_equivalent_graph({int(k):v for k,v in candidate['nodes'].items()},candidate['edges'],100)==candidate
        path=target/(stem+'-prediction.json');path.write_text(json.dumps(candidate)+'\n')
        frozen[stem]=sha(path);graphs[stem]=dict(original=original,structured=candidate)
        print(json.dumps(dict(stage='prediction_frozen',stem=stem,changed_edges=changes[stem]['changed_edges'])),flush=True)
    (target/'PRELABEL.json').write_text(json.dumps(frozen,indent=2)+'\n')
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1';manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files=json.loads(manifest.read_text())['files']
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={a:[] for a in ('original','structured')}
    for stem in stems:
        selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
        for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
        path=truth_root/'train'/(stem+'.geff')
        assert sha(target/(stem+'-prediction.json'))==frozen[stem]
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            graph=helper['prediction_graph'](graphs[stem][arm])
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            estimated=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            rows[arm].append(dict(scorer.per_sample_metrics(er,estimated,scorer.node_recall(graph,truth)),stem=stem,embryo=stem.split('_')[0]))
        print(json.dumps(dict(stage='scored',stem=stem)),flush=True)
    summaries={a:scorer.summarise(rs) for a,rs in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in rs if r['embryo']==e]) for e in ('44b6','6bba')} for a,rs in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in rs} for a,rs in rows.items()}
    for a in rows:
        assert summaries[a]['n']==10 and summaries[a]['n_adj']==10
        assert all(math.isfinite(r[k]) for r in (summaries[a],*embryos[a].values(),*movies[a].values()) for k in ('score','edge_jaccard','adj_edge_jaccard'))
    result=dict(status='independent_selection_diagnostic_complete',summaries=summaries,rows=rows,by_embryo=embryos,
        per_movie=movies,changes=changes,prediction_sha256=frozen,contract_sha256=sha(target/'CONTRACT.json'),
        delta={k:summaries['structured'][k]-summaries['original'][k] for k in ('score','edge_jaccard','adj_edge_jaccard')},
        source_movie_gate_failed=True,automatic_promotion_forbidden=True,authorized_for_submission=False,
        source_model_refit=False,selection_used_for_training=False,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    (ROOT/'reports/experiments/trajectory-structured-selection-v1-result.json').write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(helper['finite'](dict(status=result['status'],summaries=summaries,delta=result['delta'],seconds=result['seconds']))))


if __name__=='__main__':main()
