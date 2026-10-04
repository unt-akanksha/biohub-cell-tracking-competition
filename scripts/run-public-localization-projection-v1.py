"""Persist fixed localization candidates, then apply unchanged official gates."""
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,STEMS,csv_equivalent_graph
from research.public_d4_quality import compare
from research.public_localization_projection import project_graph


def main():
    started=time.monotonic()
    output=ROOT/'.biohub/cache/public-localization-projection-v1'
    target=ROOT/'reports/experiments/public-localization-projection-v1-result.json'
    if output.exists() or target.exists():raise ValueError('Preserve staged or completed experiment')
    parents=ROOT/'.biohub/cache/public-d4-full-movie-v1-output'
    prior_path=ROOT/'reports/experiments/public-d4-full-movie-v1-result.json'
    if sha(prior_path)!='04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b':
        raise ValueError('Parent official result changed')
    if sha(ROOT/'research/public_d4_quality.py')!='59b0090f8f68632faccdb7101c07ecbfc3f3a9077f96bf181bc80e595456bd36':
        raise ValueError('Frozen numerical gate changed')
    v1=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
    v2=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v2.py'))
    terminal,prepared=v1['validate_predictions'](parents,'61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    source_manifest=ROOT/'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json'
    original_manifest=json.loads(source_manifest.read_text())
    for row in original_manifest['files']:
        if row['path'].endswith('/pre-postprocess.json') and sha(parents/row['path'])!=row['sha256']:
            raise ValueError('Genuine detector reference changed')
    output.mkdir(parents=True)
    pins={name:sha(ROOT/name) for name in ('research/public_localization_projection.py',
        'scripts/run-public-localization-projection-v1.py','research/public_d4_quality.py',
        'reports/experiments/public-localization-projection-v1-design.md')}
    receipt=dict(run_id='public-localization-projection-v1',pins=pins,ground_truth_used=False,
                 radius_um=1.625,source_manifest_sha256=sha(source_manifest),records={})
    # Every candidate is generated and saved before the scoring phase starts.
    for stem in STEMS:
        receipt['records'][stem]={}
        for parent in ('original','corrected'):
            reference=json.loads((parents/f'{stem}-{parent}'/'pre-postprocess.json').read_text())
            graph,details=project_graph(prepared[stem][parent],reference)
            checked=csv_equivalent_graph({int(i):n for i,n in graph['nodes'].items()},graph['edges'],100)
            if checked!=graph:raise ValueError('Projected graph changed serialization contract')
            for ident,node in graph['nodes'].items():
                old=prepared[stem][parent]['nodes'][ident]
                if any(node[k]!=old[k] for k in ('node_id','t')):
                    raise ValueError('Node identity or time changed')
            path=output/f'{stem}-{parent}-projection.json'
            path.write_text(json.dumps(graph,sort_keys=True,allow_nan=False))
            receipt['records'][stem][parent]=dict(path=path.name,sha256=sha(path),**details)
            print(json.dumps(dict(event='candidate_persisted',stem=stem,parent=parent,**details)),flush=True)
    manifest_path=output/'prelabel-manifest.json'
    manifest_path.write_text(json.dumps(receipt,indent=2)+'\n')
    prior=json.loads(prior_path.read_text())
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    v2['verify_truth_inventory'](truth_root,'744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
    scorer=v1['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={a:[] for a in ('original_projection','corrected_projection')}
    for stem in STEMS:
        for parent in ('original','corrected'):
            record=receipt['records'][stem][parent];path=output/record['path']
            if sha(path)!=record['sha256']:raise ValueError('Persisted candidate changed')
            graph=v1['prediction_graph'](json.loads(path.read_text()))
            truth_path=truth_root/'train'/(stem+'.geff')
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            count=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(er,count,scorer.node_recall(graph,truth)),
                     stem=stem,embryo=stem.split('_')[0])
            old=next(r for r in prior['per_movie'][parent] if r['stem']==stem)
            if row['num_pred_nodes']!=old['num_pred_nodes'] or row['total_node_ratio']!=old['total_node_ratio']:
                raise ValueError('Node-count adjustment changed')
            rows[parent+'_projection'].append(row)
            print(json.dumps(v1['finite'](dict(event='scored',parent=parent,**row))),flush=True)
    summaries={a:scorer.summarise(v) for a,v in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in v if r['embryo']==e]) for e in ('44b6','6bba')}
             for a,v in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
    comparisons={}
    for arm in rows:
        comparisons[arm]=compare({'original':prior['per_movie']['original'],'corrected':rows[arm]},
            {'original':prior['summaries']['original'],'corrected':summaries[arm]},
            {'original':prior['by_embryo']['original'],'corrected':embryos[arm]},
            {'original':prior['per_movie_summaries']['original'],'corrected':movies[arm]})
    parent_delta={s:movies['corrected_projection'][s]['score']-prior['per_movie_summaries']['corrected'][s]['score'] for s in STEMS}
    extra=dict(pooled_combined_delta=summaries['corrected_projection']['score']-prior['summaries']['corrected']['score'],
               per_movie_combined_delta=parent_delta)
    extra['passed']=extra['pooled_combined_delta']>=0 and min(parent_delta.values())>=0
    comparisons['corrected_projection']['own_parent_nonregression']=extra
    comparisons['corrected_projection']['diagnostic_gate_passed'] &= extra['passed']
    if any(sha(ROOT/name)!=value for name,value in pins.items()):raise ValueError('Frozen sources changed')
    result=dict(run_id='public-localization-projection-v1',status='complete',receipt=receipt,
        manifest_sha256=sha(manifest_path),per_movie=rows,summaries=summaries,by_embryo=embryos,
        per_movie_summaries=movies,comparisons=comparisons,elapsed_seconds=time.monotonic()-started,
        gpu_hours=0,independently_held_out=False,authorized_for_submission=False,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    target.write_text(json.dumps(v1['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(v1['finite'](dict(summaries=summaries,comparisons=comparisons)),indent=2),flush=True)


if __name__=='__main__':main()
