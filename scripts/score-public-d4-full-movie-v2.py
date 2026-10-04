"""Same frozen D4 comparison; correct full-file validation for sparse truth.

V1 aborted before evaluate(): full movies need not have annotations on every
frame. Verify the historical complete GEFF file inventory instead. Predictions,
100-frame inference coverage, official scorer and numerical gates are unchanged.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,STEMS
from research.public_d4_quality import compare

V1_PATH=ROOT/'scripts/score-public-d4-full-movie-v1.py'
V1_SHA='28c630797c7298da05be40376a4de7cb920ad6cfdd773e146e44ef552a676b93'
if sha(V1_PATH)!=V1_SHA:raise ValueError('Frozen original scoring implementation changed')
V1=runpy.run_path(str(V1_PATH))


def verify_truth_inventory(cache, expected_manifest_sha):
    manifest_path=cache/'train_geff_cache_manifest.json'
    if sha(manifest_path)!=expected_manifest_sha:raise ValueError('Historical truth manifest changed')
    manifest=json.loads(manifest_path.read_text())
    selected=[]
    for stem in STEMS:
        prefix=stem+'.geff/'
        records=[r for r in manifest['files'] if r['relative_path'].startswith(prefix)]
        expected={r['relative_path'] for r in records}
        observed={p.relative_to(cache/'train').as_posix() for p in (cache/'train'/(stem+'.geff')).rglob('*') if p.is_file()}
        if len(records)!=21 or expected!=observed:
            raise ValueError('Complete original GEFF file set required')
        for row in records:
            p=cache/'train'/row['relative_path']
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
                raise ValueError('Full original annotation file changed or missing')
        selected.extend(records)
    return selected


def validate_sparse_times(times):
    if not times or any(not isinstance(t,int) or not 0<=t<100 for t in times):
        raise ValueError('Sparse annotation times outside the full image movie')


def main(args):
    if args.output.exists():raise ValueError('Refuse to overwrite completed scores')
    terminal,prepared=V1['validate_predictions'](args.predictions,args.terminal_sha256)
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    truth_receipts=verify_truth_inventory(truth_root,args.truth_manifest_sha256)
    scorer=V1['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={a:[] for a in ('original','corrected')}; coverage={}
    for stem in STEMS:
        path=truth_root/'train'/(stem+'.geff')
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            times=sorted(set(truth.node_attrs()['t'].to_list()))
            validate_sparse_times(times)
            coverage[stem]=dict(annotated_times=times,annotated_nodes=truth.num_nodes(),
                                annotated_edges=truth.num_edges(),image_frames=100,
                                full_annotation_file_inventory_verified=True)
            graph=V1['prediction_graph'](prepared[stem][arm])
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            count=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(er,count,scorer.node_recall(graph,truth)),
                     stem=stem,embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(V1['finite'](dict(event='scored',arm=arm,**row))),flush=True)
    summaries={a:scorer.summarise(r) for a,r in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in v if r['embryo']==e]) for e in ('44b6','6bba')}
             for a,v in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
    comparison=compare(rows,summaries,embryos,movies)
    result=dict(run_id='public-d4-full-movie-v1',status='scored',scoring_wrapper_version=2,
        source_sha256=sha(Path(__file__)),original_scoring_source_sha256=V1_SHA,
        comparison_source_sha256=sha(ROOT/'research/public_d4_quality.py'),
        terminal_sha256=args.terminal_sha256,contract_sha256=terminal['contract_sha256'],
        truth_manifest_sha256=args.truth_manifest_sha256,verified_truth_files=truth_receipts,
        annotation_coverage=coverage,per_movie=rows,summaries=summaries,by_embryo=embryos,
        per_movie_summaries=movies,comparison=comparison,independently_held_out=False,
        authorized_for_submission=False,authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    args.output.write_text(json.dumps(V1['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(V1['finite'](dict(summaries=summaries,comparison=comparison)),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--terminal-sha256',required=True)
    p.add_argument('--truth-manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
