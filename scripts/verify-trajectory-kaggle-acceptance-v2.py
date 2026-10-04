"""Check frozen two-T4 outputs, reuse identical scores or evaluate all eight pairs."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_division_quality_v1 import check

STEMS=('44b6_24264f12','44b6_81c256f0','6bba_23af9eeb','6bba_f1fde7e0',
       '44b6_12dfb391','44b6_267148e4','6bba_062c8d37','6bba_07e24132')
CONTRACT='d10d19b1e21b30e2eb5c6bf1c05161559f74594ce30e250e3c473f2d8b0379f2'


def pinned_json(path,expected):
    if sha(path)!=expected:raise ValueError(f'Frozen evidence changed: {path.name}')
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=Path,required=True)
    p.add_argument('--terminal-sha256',required=True)
    p.add_argument('--overlap',action='store_true')
    args=p.parse_args();start=time.monotonic()
    global CONTRACT
    if args.overlap:
        CONTRACT='851908fa5aa8ba628aad456afa94badc6ae9e10c6f9f3e36cf43147386ef8d55'
    target=(ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-result.json' if args.overlap
            else ROOT/'reports/experiments/trajectory-kaggle-acceptance-v2-result.json')
    if target.exists():raise ValueError('Preserve completed acceptance decision')
    folder=args.outputs/'trajectory-complete'
    terminal=pinned_json(folder/'result.json',args.terminal_sha256)
    if (terminal['status']!='complete' or terminal['mode']!='validation'
            or terminal['gpu_count']!=2 or terminal['contract_sha256']!=CONTRACT
            or terminal['ground_truth_opened'] or terminal['submission_created']
            or not 0<terminal['elapsed_seconds']<=2600):
        raise ValueError('Complete same-contract two-GPU validation required')
    smoke=json.loads((args.outputs/'trajectory-smoke/result.json').read_text())
    if (smoke['status']!='complete' or smoke['mode']!='smoke'
            or smoke['gpu_count']!=2 or smoke['contract_sha256']!=CONTRACT):
        raise ValueError('Same-contract two-GPU smoke missing')
    prepared={};receipts=[];times=[]
    expected_workers={'0','1','2','3'} if args.overlap else {'0','1'}
    if set(terminal['workers'])!=expected_workers:raise ValueError('All planned workers required')
    if args.overlap and (smoke.get('worker_count')!=4 or terminal.get('worker_count')!=4):
        raise ValueError('Four-worker smoke and full acceptance required')
    for shard,result in terminal['workers'].items():
        if (result['status']!='complete_prelabel_predictions' or not result['inputs_unchanged']
                or result['contract_sha256']!=CONTRACT or result['solver_backend']!='SCIP'
                or 'T4' not in result['device'] or result['ground_truth_opened']):
            raise ValueError('Incomplete or wrong GPU/backend worker')
        for stem,arms in result['movies'].items():
            if stem in prepared or stem not in STEMS or set(arms)!={'original'}:
                raise ValueError('Missing, duplicated or wrong movie')
            record=arms['original'];times.append(record['seconds'])
            if record['frames']!=100:raise ValueError('Complete100-frame validation required')
            prepared[stem]={}
            for arm,filename,field in (('original','prediction.json','prediction_sha256'),
                                       ('repaired','repaired-prediction.json','repaired_sha256')):
                path=folder/f'shard-{shard}/{stem}-original'/filename
                graph=pinned_json(path,record[field]);validate_graph(graph,100)
                prepared[stem][arm]=graph
                receipts.append(dict(movie=stem,arm=arm,path=str(path.relative_to(args.outputs)),sha256=sha(path)))
            base,new=prepared[stem]['original'],prepared[stem]['repaired']
            base_edges={(e['source_id'],e['target_id']) for e in base['edges']}
            new_edges={(e['source_id'],e['target_id']) for e in new['edges']}
            if (base['nodes']!=new['nodes'] or not base_edges<=new_edges
                    or len(new_edges-base_edges)!=record['added_edges']):
                raise ValueError('Repair changed original nodes/links')
    if set(prepared)!=set(STEMS):raise ValueError('All eight movies required before truth access')
    if sha(folder/'validation-predictions.csv')!=terminal['csv']['sha256']:
        raise ValueError('Assembled validation CSV changed')
    prior=pinned_json(ROOT/'reports/experiments/trajectory-division-full-movie-v1-result.json',
                       '6dc733376a89e0eb8fdb8b996caf008252c84799199d35ef5f7f48a6ff39bb40')
    base_terminal=pinned_json(ROOT/'.biohub/cache/public-d4-full-movie-v1-output/result.json',
                       '61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    mixture=pinned_json(ROOT/'reports/experiments/trajectory-source-mixture-v1-diagnostic.json',
                       '4216fa5bb693957806eaa04e25f3c0b297e5507674bac754f649edca6e1dd6f3')
    additional=pinned_json(ROOT/'.biohub/cache/trajectory-division-full-v1-output/result.json',
                       '40c35432a298d22f347cbdd111b1cd6d21ea3965acc35b4fc63369d3147544b1')
    identical={}
    for stem in STEMS:
        refs={}
        if stem in STEMS[:4]:
            path=ROOT/f'.biohub/cache/public-d4-full-movie-v1-output/{stem}-original/prediction.json'
            refs['original']=pinned_json(path,base_terminal['movies'][stem]['original']['prediction_sha256'])
            path=ROOT/f'.biohub/cache/trajectory-source-mixture-v1-diagnostic/{stem}.json'
            refs['repaired']=pinned_json(path,mixture['receipts'][stem]['sha256'])
        else:
            for arm,filename,field in (('original','prediction.json','prediction_sha256'),
                                      ('repaired','repaired-prediction.json','repaired_sha256')):
                path=ROOT/f'.biohub/cache/trajectory-division-full-v1-output/{stem}-original/{filename}'
                refs[arm]=pinned_json(path,additional['movies'][stem]['original'][field])
        identical[stem]={arm:prepared[stem][arm]==refs[arm] for arm in refs}
    exact=all(all(v.values()) for v in identical.values())
    ground_truth_opened=False
    if exact:
        rows=prior['joint_eight_movie_rows'];summaries=prior['summaries']
        embryos=prior['by_embryo'];movies=prior['per_movie_summaries'];comparison=prior['comparison']
    else:
        # All sixteen predictions were already frozen and validated above.
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
        manifest=pinned_json(truth_root/'train_geff_cache_manifest.json',
                            '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
        for stem in STEMS:
            records=[r for r in manifest['files'] if r['relative_path'].startswith(stem+'.geff/')]
            observed={p.relative_to(truth_root/'train').as_posix()
                      for p in (truth_root/'train'/(stem+'.geff')).rglob('*') if p.is_file()}
            if len(records)!=21 or observed!={r['relative_path'] for r in records}:
                raise ValueError('Complete original truth inventory required')
            for record in records:
                path=truth_root/'train'/record['relative_path']
                if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                    raise ValueError('Ground truth changed')
        wrapper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
        scorer=wrapper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        ground_truth_opened=True;rows={a:[] for a in ('original','repaired')}
        for stem in STEMS:
            path=truth_root/'train'/(stem+'.geff')
            truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            count=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            for arm in rows:
                graph=wrapper['prediction_graph'](prepared[stem][arm])
                result=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
                row=dict(scorer.per_sample_metrics(result,count,scorer.node_recall(graph,truth)),
                         stem=stem,embryo=stem.split('_')[0])
                rows[arm].append(row)
                print(json.dumps(wrapper['finite'](dict(event='frozen_movie_scored',arm=arm,**row))),flush=True)
        summaries={a:scorer.summarise(v) for a,v in rows.items()}
        embryos={a:{e:scorer.summarise([r for r in v if r['embryo']==e]) for e in ('44b6','6bba')} for a,v in rows.items()}
        movies={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
        comparison=check(rows,summaries,embryos,movies,STEMS[4:])
    result=dict(status='acceptance_passed' if comparison['diagnostic_gate_passed'] else 'rejected_runtime_quality',
        contract_sha256=CONTRACT,terminal_sha256=args.terminal_sha256,verified_graphs=receipts,
        exact_A10_graph_identity=identical,score_reused_without_reopening_truth=exact,
        ground_truth_opened=ground_truth_opened,rows=rows,summaries=summaries,by_embryo=embryos,
        per_movie_summaries=movies,comparison=comparison,
        runtime=dict(two_t4_verified=True,worker_count=len(expected_workers),
                     wall_seconds=terminal['elapsed_seconds'],movie_seconds=times,
                     mean_movie_seconds=sum(times)/8,maximum_movie_seconds=max(times),
                     projected_199_movies_two_gpus_hours=(terminal['elapsed_seconds']/8*199/3600
                         if args.overlap else sum(times)/8*199/2/3600),
                     projection_method='complete_cohort_makespan' if args.overlap else 'mean_movie_two_workers',
                     projection_is_not_a_hidden_runtime_guarantee=True),
        production_notebook_eligible=comparison['diagnostic_gate_passed'],
        independently_held_out=False,public_backbone_training_overlap=True,
        submission_performed=False,elapsed_seconds=time.monotonic()-start)
    if not args.overlap:
        result['runtime']['projected_199_movies_two_workers_hours']=sum(times)/8*199/2/3600
    wrapper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
    result=wrapper['finite'](result)
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('status','summaries','comparison','runtime')},indent=2),flush=True)


if __name__=='__main__':main()
