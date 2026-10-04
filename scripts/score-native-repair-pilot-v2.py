"""Hash-verified full-movie paired official scoring, never partial smoke scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import validate_graph


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,default=ROOT/'.biohub/cache/native-repair-pilot-v2-bundle')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve completed score')
    begin=time.monotonic();bundle=args.bundle
    contract=json.loads((bundle/'PILOT.json').read_text());terminal=json.loads((args.predictions/'RESULT.json').read_text())
    if terminal['status']!='complete_prelabel_predictions' or terminal['smoke'] or terminal['ground_truth_opened'] or terminal['pilot_sha256']!=sha(bundle/'PILOT.json'):
        raise ValueError('Require exact complete, prelabel, non-smoke pilot')
    for record in contract['files']:
        if sha(bundle/record['path'])!=record['sha256']:raise ValueError('Frozen pilot code/input changed')
    if [m['stem'] for m in terminal['movies']]!=[m['stem'] for m in contract['movies']]:raise ValueError('Incomplete movie set')
    prepared={}
    for expected,record in zip(contract['movies'],terminal['movies']):
        stem=record['stem'];parent=bundle/expected['graph'];prediction=args.predictions/(stem+'.json')
        repairs=args.predictions/(stem+'-repairs.json')
        if record['frames_evaluated']!=100 or sha(parent)!=record['parent_sha256'] or sha(prediction)!=record['prediction_sha256'] or sha(repairs)!=record['repairs_sha256']:
            raise ValueError('Incomplete or changed prediction artifacts')
        original=json.loads(parent.read_text());candidate=json.loads(prediction.read_text())
        validate_graph(original,100);validate_graph(candidate,100)
        if candidate['nodes']!=original['nodes'] or record['groups']!=sum(n['t']>0 for n in original['nodes'].values()):
            raise ValueError('Changed nodes or incomplete query coverage')
        prepared[stem]=dict(original=original,candidate=candidate)
    neutral=all({(e['source_id'],e['target_id']) for e in graphs['original']['edges']}==
                {(e['source_id'],e['target_id']) for e in graphs['candidate']['edges']} for graphs in prepared.values())
    if neutral:
        result=dict(run_id=contract['run_id'],status='verified_neutral_complete_graphs',
            movies=list(prepared),all_nodes_and_edges_identical=True,ground_truth_opened=False,
            pilot_gate_passed=False,authorized_for_submission=False,
            terminal_sha256=sha(args.predictions/'RESULT.json'),pilot_sha256=sha(bundle/'PILOT.json'),
            elapsed_seconds=time.monotonic()-begin,reason='No semantic graph changes; official rescoring cannot produce a genuine gain')
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    # Only now verify and open the exact complete sparse-annotation GEFFs.
    geffs=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=geffs/'train_geff_cache_manifest.json'
    if sha(manifest)!='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':raise ValueError('Official labels changed')
    for stem in prepared:
        records=[r for r in json.loads(manifest.read_text())['files'] if r['relative_path'].startswith(stem+'.geff/')]
        if len(records)!=21:raise ValueError('Incomplete GEFF file set')
        for r in records:
            path=geffs/'train'/r['relative_path']
            if path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:raise ValueError('Changed GEFF artifact')
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={arm:[] for arm in ('original','candidate')}
    for stem,graphs in prepared.items():
        truth_path=geffs/'train'/(stem+'.geff')
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            graph=helper['prediction_graph'](graphs[arm])
            evaluation=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            count=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(evaluation,count,scorer.node_recall(graph,truth)),stem=stem,embryo=stem.split('_')[0])
            rows[arm].append(row);print(json.dumps(helper['finite'](dict(arm=arm,stem=stem,summary=scorer.summarise([row])))),flush=True)
    summaries={a:scorer.summarise(r) for a,r in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in rs if r['embryo']==e]) for e in ('44b6','6bba')} for a,rs in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in rs} for a,rs in rows.items()}
    delta={k:summaries['candidate'][k]-summaries['original'][k] for k in ('score','edge_jaccard','adj_edge_jaccard')}
    by_embryo={e:embryos['candidate'][e]['score']-embryos['original'][e]['score'] for e in ('44b6','6bba')}
    by_movie={s:movies['candidate'][s]['score']-movies['original'][s]['score'] for s in prepared}
    if not all(math.isfinite(v) for v in [*delta.values(),*by_embryo.values(),*by_movie.values()]):raise ValueError('Nonfinite paired metric')
    gates=dict(pooled_strict_improvement=delta['score']>0,raw_edges_strict_improvement=delta['edge_jaccard']>0,
               embryo_nonregression=min(by_embryo.values())>=0,movie_nonregression=min(by_movie.values())>=0)
    result=dict(run_id=contract['run_id'],status='scored',per_movie=rows,summaries=summaries,by_embryo=embryos,
        per_movie_summaries=movies,delta=delta,embryo_delta=by_embryo,movie_delta=by_movie,gates=gates,pilot_gate_passed=all(gates.values()),
        worst_movies={a:min(movies[a],key=lambda s:movies[a][s]['score']) for a in rows},
        terminal_sha256=sha(args.predictions/'RESULT.json'),pilot_sha256=sha(bundle/'PILOT.json'),source_sha256=sha(Path(__file__)),
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        independently_held_out=False,limitation='Public base checkpoint overlap; all eight held-out movies/runtime still required',
        authorized_for_submission=False,elapsed_seconds=time.monotonic()-begin)
    args.output.write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(helper['finite']({k:result[k] for k in ['summaries','delta','gates','pilot_gate_passed','elapsed_seconds']})),flush=True)


if __name__=='__main__':main()
