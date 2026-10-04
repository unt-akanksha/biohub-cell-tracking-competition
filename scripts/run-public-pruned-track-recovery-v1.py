"""CPU-only reconstruction and evidence-gated recovery of genuine tracklets."""
import collections
import json
import math
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,STEMS,csv_equivalent_graph
from research.public_d4_preflight import named_definitions
from research.public_d4_quality import compare
from research.public_pruned_track_recovery import recover


def main():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    started=time.monotonic()
    output=ROOT/'.biohub/cache/public-pruned-track-recovery-v1'
    target=ROOT/'reports/experiments/public-pruned-track-recovery-v1-result.json'
    if output.exists() or target.exists():raise ValueError('Preserve a staged or completed run')
    parents=ROOT/'.biohub/cache/public-d4-full-movie-v1-output'
    prior_path=ROOT/'reports/experiments/public-d4-full-movie-v1-result.json'
    if sha(prior_path)!='04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b':
        raise ValueError('Parent metric evidence changed')
    v1=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
    v2=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v2.py'))
    _,prepared=v1['validate_predictions'](parents,'61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    artifacts=json.loads((ROOT/'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    for row in artifacts['files']:
        if row['path'].endswith(('/pre-postprocess.json','/postprocess-stats.json')) and sha(parents/row['path'])!=row['sha256']:
            raise ValueError('Frozen inference artifact changed')
    source_path=ROOT/'.biohub/cache/public-d4-correction-v1/public-postprocess-d4-corrected.py'
    if sha(source_path)!='f5122587381f17a451547327abff3948734e455a2ae8cfc9d09e0e506c1e974d':
        raise ValueError('Public postprocess source changed')
    config_path=ROOT/'.biohub/cache/public-d4-full-movie-v1-bundle/resolved-public-config.json'
    contract=json.loads((ROOT/'.biohub/cache/public-d4-full-movie-v1-bundle/CONTRACT.json').read_text())
    if sha(config_path)!=contract['bundle_sha256']['resolved-public-config.json']:
        raise ValueError('Frozen public configuration changed')
    config=json.loads(config_path.read_text())['globals']
    env=dict(config,np=np,math=math,linear_sum_assignment=linear_sum_assignment,
             VOXEL_SCALE_UM=(1.625,.40625,.40625))
    exec(compile(named_definitions(source_path.read_text(),('edge_distance_um','_position_um',
                 'motion_relink_edges','linefit_smooth_output_graph')),'frozen-public-motion-and-smoothing','exec'),env)
    output.mkdir(parents=True)
    pins={n:sha(ROOT/n) for n in ('research/public_pruned_track_recovery.py',
        'scripts/run-public-pruned-track-recovery-v1.py','research/public_d4_quality.py',
        'reports/experiments/public-pruned-track-recovery-v1-design.md')}
    receipt=dict(run_id='public-pruned-track-recovery-v1',pins=pins,records={},ground_truth_used=False)
    for stem in STEMS:
        receipt['records'][stem]={}
        for parent in ('original','corrected'):
            t0=time.monotonic()
            print(json.dumps(dict(event='reconstructing_motion',stem=stem,parent=parent)),flush=True)
            source=parents/f'{stem}-{parent}'
            pre=json.loads((source/'pre-postprocess.json').read_text())
            saved_stats=json.loads((source/'postprocess-stats.json').read_text())
            nodes={int(i):n for i,n in pre['nodes'].items()};probs={}
            for edge in pre['edges']:
                a,b=int(edge['source_id']),int(edge['target_id'])
                if nodes[b]['t']!=nodes[a]['t']+1:continue
                if env['edge_distance_um'](nodes[a],nodes[b])>config['OUTPUT_EDGE_MAX_UM']:continue
                probability=float(edge['edge_prob'])
                if np.isfinite(probability):probs[a,b]=max(probs.get((a,b),-float('inf')),probability)
            stats=collections.defaultdict(int)
            motion=env['motion_relink_edges'](nodes,stats,probs)
            count_keys=('motion_relink_edges','motion_relink_frames','motion_relink_tight_edges','motion_relink_relaxed_edges')
            if any(stats[k]!=saved_stats[k] for k in count_keys):
                raise ValueError('Motion reconstruction differs from saved full-run counts')
            (output/f'{stem}-{parent}-motion.json').write_text(json.dumps(motion,sort_keys=True,allow_nan=False))
            base=prepared[stem][parent]
            graph,details=recover(base,pre,motion,config,env['linefit_smooth_output_graph'],
                                  len(base['nodes'])+saved_stats['short_track_nodes_removed'])
            if csv_equivalent_graph({int(i):n for i,n in graph['nodes'].items()},graph['edges'],100)!=graph:
                raise ValueError('Recovered graph violates complete output contract')
            path=output/f'{stem}-{parent}-recovery.json'
            path.write_text(json.dumps(graph,sort_keys=True,allow_nan=False))
            row=dict(path=path.name,sha256=sha(path),motion_counts={k:stats[k] for k in count_keys},
                     seconds=time.monotonic()-t0,**details)
            receipt['records'][stem][parent]=row
            print(json.dumps(dict(event='candidate_persisted',stem=stem,parent=parent,**row)),flush=True)
    manifest=output/'prelabel-manifest.json'
    manifest.write_text(json.dumps(receipt,indent=2)+'\n')
    # All eight candidate graphs now exist and are frozen before scoring.
    prior=json.loads(prior_path.read_text())
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    v2['verify_truth_inventory'](truth_root,'744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
    scorer=v1['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows={a:[] for a in ('original_recovery','corrected_recovery')}
    for stem in STEMS:
        for parent in ('original','corrected'):
            record=receipt['records'][stem][parent];path=output/record['path']
            if sha(path)!=record['sha256']:raise ValueError('Frozen candidate changed')
            graph=v1['prediction_graph'](json.loads(path.read_text()))
            truth_path=truth_root/'train'/(stem+'.geff')
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            count=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(er,count,scorer.node_recall(graph,truth)),
                     stem=stem,embryo=stem.split('_')[0])
            rows[parent+'_recovery'].append(row)
            print(json.dumps(v1['finite'](dict(event='scored',parent=parent,**row))),flush=True)
    summaries={a:scorer.summarise(v) for a,v in rows.items()}
    embryos={a:{e:scorer.summarise([r for r in v if r['embryo']==e]) for e in ('44b6','6bba')}
             for a,v in rows.items()}
    movies={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
    comparisons={}
    for parent in ('original','corrected'):
        arm=parent+'_recovery'
        comparisons[arm]=compare({'original':prior['per_movie']['original'],'corrected':rows[arm]},
            {'original':prior['summaries']['original'],'corrected':summaries[arm]},
            {'original':prior['by_embryo']['original'],'corrected':embryos[arm]},
            {'original':prior['per_movie_summaries']['original'],'corrected':movies[arm]})
        tp_gain=sum(r['edge_tp'] for r in rows[arm])-sum(r['edge_tp'] for r in prior['per_movie'][parent])
        comparisons[arm]['added_edge_tp_against_parent']=tp_gain
        comparisons[arm]['diagnostic_gate_passed'] &= tp_gain>0
    parent_delta={s:movies['corrected_recovery'][s]['score']-prior['per_movie_summaries']['corrected'][s]['score'] for s in STEMS}
    extra=dict(pooled_combined_delta=summaries['corrected_recovery']['score']-prior['summaries']['corrected']['score'],
               per_movie_combined_delta=parent_delta)
    extra['passed']=extra['pooled_combined_delta']>=0 and min(parent_delta.values())>=0
    comparisons['corrected_recovery']['own_parent_nonregression']=extra
    comparisons['corrected_recovery']['diagnostic_gate_passed'] &= extra['passed']
    if any(sha(ROOT/n)!=digest for n,digest in pins.items()):raise ValueError('Frozen scientific code changed')
    result=dict(run_id='public-pruned-track-recovery-v1',status='complete',receipt=receipt,
        manifest_sha256=sha(manifest),per_movie=rows,summaries=summaries,by_embryo=embryos,
        per_movie_summaries=movies,comparisons=comparisons,elapsed_seconds=time.monotonic()-started,
        gpu_hours=0,independently_held_out=False,authorized_for_submission=False,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    target.write_text(json.dumps(v1['finite'](result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(v1['finite'](dict(summaries=summaries,comparisons=comparisons)),indent=2),flush=True)


if __name__=='__main__':main()
