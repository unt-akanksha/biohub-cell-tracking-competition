"""Frozen T4-cache eight-movie quality test; no fitting or source-gate erasure."""
import json
import math
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,csv_equivalent_graph
from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_candidate_ranker_v1 import features
from research.trajectory_disagreement_data_v1 import extract
from research.trajectory_joint_assignment_v1 import apply


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-structured-eight-v1';assert not target.exists()
    accepted_path=ROOT/'reports/experiments/trajectory-overlap-cache-kaggle-v2-result.json'
    assert sha(accepted_path)=='2934de829c237906f214f7c6e4583b961b6e59aeb73c8a8adde05c52d69a1ede'
    accepted=json.loads(accepted_path.read_text());assert accepted['status']=='acceptance_passed'
    base=ROOT/'.biohub/cache/trajectory-overlap-cache-kaggle-v1-output'
    assert sha(base/'trajectory-complete/result.json')==accepted['terminal_sha256']
    paths={}
    for record in accepted['verified_graphs']:
        p=base/record['path'];assert sha(p)==record['sha256']
        if record['arm']=='repaired':paths[record['movie']]=p
    assert len(paths)==8
    selection_path=ROOT/'reports/experiments/trajectory-structured-selection-v1-result.json'
    selection=json.loads(selection_path.read_text())
    assert selection['status']=='independent_selection_diagnostic_complete'
    assert selection['delta']['score']>0 and selection['delta']['edge_jaccard']>0
    assert all(selection['per_movie']['structured'][s]['score']>=r['score'] for s,r in selection['per_movie']['original'].items())
    model=ROOT/'.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
    assert sha(model)=='e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
    with np.load(model,allow_pickle=False) as f:weights=f['6bba']
    target.mkdir();prepared={};records={};details={}
    contract=dict(model_sha256=sha(model),source_sha256=sha(Path(__file__)),selection_result_sha256=sha(selection_path),
                  cached_T4_acceptance_sha256=sha(accepted_path),source_movie_gate_failed=True,
                  fit_or_threshold_changes=False,authorized_for_submission=False)
    (target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    for stem,path in paths.items():
        folder=path.parent;initial_path=folder/'pre-postprocess.json';raw_path=folder/'raw-candidates.npz'
        initial=json.loads(initial_path.read_text());original=json.loads(path.read_text())
        with np.load(raw_path,allow_pickle=False) as raw:
            # Functional identity guard for previously downloaded raw/intermediate data.
            extract(initial,original,raw['coords'],raw['edges'])
            groups=candidates(initial,original);matrix=features(initial,original,groups,raw['edges'])
        candidate,details[stem]=apply(original,groups,matrix,weights)
        assert csv_equivalent_graph({int(k):v for k,v in candidate['nodes'].items()},candidate['edges'],100)==candidate
        out=target/(stem+'-prediction.json');out.write_text(json.dumps(candidate)+'\n')
        records[stem]=dict(baseline_sha256=sha(path),initial_sha256=sha(initial_path),raw_sha256=sha(raw_path),prediction_sha256=sha(out))
        prepared[stem]=dict(original=original,structured=candidate)
        print(json.dumps(dict(stage='prediction_frozen',stem=stem,changed_edges=details[stem]['changed_edges'])),flush=True)
    (target/'PRELABEL.json').write_text(json.dumps(records,indent=2)+'\n')
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1';manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    files=json.loads(manifest.read_text())['files']
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    reference={r['stem']:r for r in accepted['rows']['repaired']}
    rows={a:[] for a in ('original','structured')}
    for stem in paths:
        selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
        for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
        path=truth_root/'train'/(stem+'.geff')
        assert sha(target/(stem+'-prediction.json'))==records[stem]['prediction_sha256']
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            graph=helper['prediction_graph'](prepared[stem][arm])
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            estimated=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(er,estimated,scorer.node_recall(graph,truth)),stem=stem,embryo=stem.split('_')[0])
            if arm=='original':
                for k in scorer.COUNT_COLUMNS:assert row[k]==reference[stem][k]
            rows[arm].append(row)
        print(json.dumps(dict(stage='scored',stem=stem)),flush=True)
    summaries={a:scorer.summarise(rs) for a,rs in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in rs if r['embryo']==e]) for e in ('44b6','6bba')} for a,rs in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in rs} for a,rs in rows.items()}
    for a in rows:
        assert summaries[a]['n']==8 and summaries[a]['n_adj']==8
        assert all(math.isfinite(r[k]) for r in (summaries[a],*embryos[a].values(),*movies[a].values()) for k in ('score','edge_jaccard','adj_edge_jaccard'))
    gates=dict(pooled_strict=summaries['structured']['score']>summaries['original']['score'],
        raw_edges_strict=summaries['structured']['edge_jaccard']>summaries['original']['edge_jaccard'],
        embryo_nonregression=all(embryos['structured'][e]['score']>=embryos['original'][e]['score'] for e in embryos['original']),
        movie_nonregression=all(movies['structured'][s]['score']>=movies['original'][s]['score'] for s in paths))
    result=dict(status='complete_eight_movie_validation',summaries=summaries,rows=rows,by_embryo=embryos,per_movie=movies,
        records=records,changes=details,gates=gates,quality_pass=all(gates.values()),
        contract_sha256=sha(target/'CONTRACT.json'),source_movie_gate_failed=True,source_failure_not_reclassified=True,
        independently_held_out_from_public_backbone=False,model_fit_on_these_movies=False,
        runtime_reexecution_required=True,authorized_for_submission=False,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    (ROOT/'reports/experiments/trajectory-structured-eight-v1-result.json').write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(helper['finite'](dict(status=result['status'],summaries=summaries,gates=gates,seconds=result['seconds']))))


if __name__=='__main__':main()
