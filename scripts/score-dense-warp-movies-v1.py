"""Score every frozen full movie/arm with the patched official metric."""
import json
from pathlib import Path
import runpy
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,csv_equivalent_graph
from research.dense_warp_movie_quality_v1 import compare


def main():
    output=ROOT/'reports/experiments/dense-warp-movies-v1-score.json';assert not output.exists()
    started=time.monotonic();bundle=ROOT/'.biohub/cache/dense-warp-movie-v1-bundle'
    predictions=ROOT/'.biohub/cache/dense-warp-movie-v1-full-output'
    contract=json.loads((bundle/'CONTRACT.json').read_text());terminal=json.loads((predictions/'result.json').read_text())
    assert terminal['status']=='complete_prelabel_predictions' and terminal['mode']=='full'
    assert terminal['contract_sha256']==sha(bundle/'CONTRACT.json') and terminal['inputs_unchanged']
    assert not terminal['ground_truth_opened'] and not terminal['authorized_for_submission']
    assert 0<terminal['elapsed_seconds']<=3600
    assert sorted(terminal['movies'])==sorted(contract['stems'])
    for name,digest in contract['bundle_sha256'].items():assert sha(bundle/name)==digest
    receipt=json.loads((ROOT/'reports/experiments/dense-warp-movie-v1-full-harvest.json').read_text())
    assert receipt['status']=='verified_backup'
    for row in receipt['records']:assert sha(predictions/row['path'])==row['sha256']
    prepared={}
    for stem in contract['stems']:
        assert set(terminal['movies'][stem])==set(contract['arms'])
        prepared[stem]={}
        for arm in contract['arms']:
            row=terminal['movies'][stem][arm];assert row['frames']==100
            path=predictions/(stem+'-'+arm)/'repaired-prediction.json'
            assert sha(path)==row['repaired_sha256']
            graph=json.loads(path.read_text())
            assert csv_equivalent_graph({int(k):v for k,v in graph['nodes'].items()},graph['edges'],100)==graph
            prepared[stem][arm]=graph
        old=json.loads((bundle/(stem+'-control.json')).read_text());current=prepared[stem]['original']
        assert old['nodes']==current['nodes']
        assert {(e['source_id'],e['target_id']) for e in old['edges']}=={(e['source_id'],e['target_id']) for e in current['edges']}
    # No labels are opened until all three arms are complete and immutable.
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    for stem in prepared:
        records=[r for r in json.loads(manifest.read_text())['files'] if r['relative_path'].startswith(stem+'.geff/')]
        assert len(records)==21
        for row in records:assert sha(truth_root/'train'/row['relative_path'])==row['sha256']
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={a:[] for a in contract['arms']}
    for stem,graphs in prepared.items():
        truth_path=truth_root/'train'/(stem+'.geff')
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            graph=helper['prediction_graph'](graphs[arm])
            evaluation=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            count=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(evaluation,count,scorer.node_recall(graph,truth)),stem=stem,embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(helper['finite'](dict(stem=stem,arm=arm,summary=scorer.summarise([row])))),flush=True)
    summaries={a:scorer.summarise(rs) for a,rs in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in rs if r['embryo']==e]) for e in ('44b6','6bba')} for a,rs in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in rs} for a,rs in rows.items()}
    comparisons=compare(summaries,embryos,movies)
    result=dict(status='scored_complete_movies',rows=rows,summaries=summaries,by_embryo=embryos,
                per_movie_summaries=movies,comparisons=comparisons,
                worst_movies={a:min(ms,key=lambda s:ms[s]['score']) for a,ms in movies.items()},
                terminal_sha256=sha(predictions/'result.json'),contract_sha256=sha(bundle/'CONTRACT.json'),
                scorer_runner_sha256=sha(Path(__file__)),quality_helper_sha256=sha(ROOT/'research/dense_warp_movie_quality_v1.py'),
                authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
                partial_screen_failures_preserved=True,public_backbone_training_overlap=True,independently_held_out=False,
                all_eight_movies_validated=False,authorized_for_submission=False,seconds=time.monotonic()-started)
    output.write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(helper['finite'](dict(summaries=summaries,comparisons=comparisons))),flush=True)


if __name__=='__main__':main()
