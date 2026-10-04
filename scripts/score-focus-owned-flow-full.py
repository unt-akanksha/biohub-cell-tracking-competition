"""Verify all eight graphs and motion caches before complete-movie CPU scoring."""
import ast
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='focus-owned-flow-full-v1'
G=runpy.run_path(str(ROOT/'research/focus_owned_flow_full_contract.py'))
INFER=runpy.run_path(str(ROOT/'scripts/run-focus-owned-flow-full.py'))
SCORER=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))


def compare(rows,summaries,by_embryo):
    current,base=rows['candidate'],rows['control']
    if any([r['stem'] for r in values]!=G['STEMS'] for values in (current,base)):
        raise ValueError('Exact four-movie paired scope required')
    if any(c['num_pred_nodes']!=b['num_pred_nodes'] or c['node_recall']!=b['node_recall'] for c,b in zip(current,base)):
        raise ValueError('Same raw detections and detection recall required')
    fields=('score','edge_jaccard','node_recall')
    if any(not math.isfinite(summaries[a][k]) for a in summaries for k in fields):
        raise ValueError('Finite complete-movie aggregate metrics required')
    deltas={k:summaries['candidate'][k]-summaries['control'][k] for k in fields}
    movies=[dict(stem=c['stem'],adjusted_edge_delta=c['adj_edge_jaccard']-b['adj_edge_jaccard']) for c,b in zip(current,base)]
    if any(not math.isfinite(r['adjusted_edge_delta']) for r in movies):
        raise ValueError('Finite paired movie metrics required')
    embryo_delta={e:by_embryo['candidate'][e]['score']-by_embryo['control'][e]['score'] for e in ('44b6','6bba')}
    if any(not math.isfinite(v) for v in embryo_delta.values()): raise ValueError('Finite embryo metrics required')
    worst=min(r['adj_edge_jaccard'] for r in current)-min(r['adj_edge_jaccard'] for r in base)
    conditions=dict(score_gain=deltas['score']>0,raw_edge_gain=deltas['edge_jaccard']>0,
        recall_identical=deltas['node_recall']==0.,three_movies_improve=sum(r['adjusted_edge_delta']>0 for r in movies)>=3,
        neither_embryo_regresses=min(embryo_delta.values())>=0,
        per_movie_loss_bounded=min(r['adjusted_edge_delta'] for r in movies)>=-.02-1e-12,
        worst_movie_preserved=worst>=-.01-1e-12)
    return dict(summary_deltas=deltas,per_movie_deltas=movies,per_embryo_score_deltas=embryo_delta,
        worst_movie_delta=worst,diagnostic_conditions=conditions,diagnostic_gate_passed=all(conditions.values()))


def prepare(folder,notebook):
    import numpy as np
    import tracksdata as td
    from research.independent_motion_prior import link_motion
    from research.backward_flow_linking import link_backward_flow
    nb=json.loads(notebook.read_text()); source=''.join(nb['cells'][1]['source'])
    for name,file,directory in (('sources','source_hashes.json','repo'),('runtime_sources','runtime_hashes.json','runtime')):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        bundle=ast.literal_eval(node.value)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in bundle.items()}
        if json.loads((folder/file).read_text())!=expected: raise ValueError('Executed source manifest changed')
        for path,digest in expected.items():
            if hashlib.sha256((folder/directory/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Executed source bytes changed')
    policy=G['receipt']((folder/'runtime/verified_probe.json').read_bytes(),(folder/'runtime/split.json').read_bytes())
    if nb['metadata']['codex']['contract']!=policy: raise ValueError('Frozen full diagnostic contract changed')
    raw_root=ROOT/'.biohub/cache/kernel-outputs/focus3d-raw-detections-v1'
    raw=runpy.run_path(str(ROOT/'scripts/verify-focus-raw-detections.py'))['verify'](raw_root)
    if raw['terminal_sha256']!=policy['raw_terminal_sha256']: raise ValueError('Raw cache changed')
    manifest=json.loads((folder/'outputs/flow_manifest.json').read_text())
    terminal=json.loads((folder/'launcher_terminal.json').read_text())
    if (terminal['status']!='completed' or terminal['run_id']!=RUN or terminal['declared_budget_seconds']!=3600
        or not 0<terminal['elapsed_seconds']<=3600 or terminal['submission_performed'] is not False
        or manifest['status']!='completed' or manifest['run_id']!=RUN or manifest['contract']!=policy
        or manifest['probe_replayed'] is not True or manifest['ground_truth_opened'] is not False
        or manifest['authorized_for_submission'] is not False or manifest['new_target_movies_opened']!=0
        or manifest['frozen_flow_before']!=policy['flow_tensor_sha256'] or manifest['frozen_flow_after']!=policy['flow_tensor_sha256']
        or [r['stem'] for r in manifest['records']]!=G['STEMS']):
        raise ValueError('Complete bounded immutable four-movie inference required')
    probe_path=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-probe-v1/focus_owned_flow_probe/outputs/sampled_flow.npz'
    if hashlib.sha256(probe_path.read_bytes()).hexdigest()!=policy['probe_motion_sha256']:
        raise ValueError('Original small probe motion changed')
    with np.load(probe_path,allow_pickle=False) as data:
        probe_coords=data['coords'].copy(); probe_flow=data['backward_um'].copy()
    prepared={}; replay_count=0
    for record,original in zip(manifest['records'],raw['movies']):
        stem=record['stem']; movie=folder/'outputs'/stem
        if (original['stem']!=stem or record['image_shape']!=[100,64,256,256]
            or record['processed_frames']!=100 or record['processed_pairs']!=99
            or record['all_nodes_covered'] is not True or record['raw_checkpoint_sha256']!=original['sha256']):
            raise ValueError('Complete exact raw movie scope required')
        with np.load(raw_root/'raw_detections'/(stem+'.npz'),allow_pickle=False) as data: coords=data['coords'].copy()
        sample=movie/'sampled_flow.npz'
        if hashlib.sha256(sample.read_bytes()).hexdigest()!=record['sample_sha256']:
            raise ValueError('Full motion checksum mismatch')
        with np.load(sample,allow_pickle=False) as data:
            if set(data.files)!={'coords','backward_um'} or not np.array_equal(data['coords'],coords):
                raise ValueError('Motion cache changed raw coordinates/order')
            flow=data['backward_um'].copy()
        if (flow.shape!=(len(coords),3) or not np.isfinite(flow).all() or (flow[coords[:,0]==0]!=0).any()
            or hashlib.sha256(coords.tobytes()).hexdigest()!=record['coordinate_sha256']):
            raise ValueError('Invalid aligned physical motion')
        if len(record['sampler_receipts'])!=99: raise ValueError('Missing sampled frames')
        for t,row in enumerate(record['sampler_receipts'],1):
            points=coords[coords[:,0]==t,1:]
            if (row['frame']!=t or row['sampled_nodes']!=len(points)
                or row['trailing_border_extended_nodes']!=int(np.any(points/[1,4,4]>[63,63,63],axis=1).sum())
                or row['policy']!=policy['sampling'] or row['coordinates_modified'] is not False or row['nodes_deleted'] is not False):
                raise ValueError('Incorrect full boundary/coverage receipt')
        if stem==policy['probe_stem']:
            if (record['probe_replayed'] is not True or not np.array_equal(coords[coords[:,0]<3],probe_coords)
                or not np.array_equal(flow[coords[:,0]<3],probe_flow)):
                raise ValueError('Actual full-run/small-probe replay mismatch')
            replay_count+=1
        elif record['probe_replayed'] is not None: raise ValueError('Unexpected probe movie')
        expected_edges={'control':link_motion(coords),'candidate':link_backward_flow(coords,flow)}
        arms={}
        for arm,edges in expected_edges.items():
            path=movie/(arm+'.geff')
            if INFER['tree_hash'](path)!=record['graphs'][arm]['graph_sha256']: raise ValueError('Graph checksum mismatch')
            graph=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            nodes=graph.node_attrs().sort('node_id'); mapping={v:i for i,v in enumerate(nodes['node_id'])}
            if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords): raise ValueError('Graph changed raw nodes')
            columns=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
            actual=[] if graph.num_edges()==0 else sorted((mapping[s],mapping[t]) for s,t in graph.edge_attrs(attr_keys=columns).select(columns).iter_rows())
            if actual!=sorted((s,t) for s,t,p in edges): raise ValueError('Graph differs from frozen full linking replay')
            if record['graphs'][arm]['nodes']!=len(coords) or record['graphs'][arm]['edges']!=len(edges):
                raise ValueError('Graph counts differ from artifact')
            arms[arm]=path
        prepared[stem]=arms
    if replay_count!=1: raise ValueError('One actual probe replay required')
    return prepared,policy,manifest


def score(folder,notebook,truth_root):
    prepared,policy,manifest=prepare(folder,notebook)
    import tracksdata as td
    from geff import GeffMetadata
    metric=SCORER['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    rows={'control':[],'candidate':[]}
    for stem,arms in prepared.items():
        for arm,path in arms.items():
            gt_path=truth_root/(stem+'.geff')
            truth=td.graph.IndexedRXGraph.from_geff(str(gt_path))[0]
            graph=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            total=float(GeffMetadata.read(str(gt_path)).extra['estimated_number_of_nodes'])
            row=dict(metric.per_sample_metrics(er,total,SCORER['diagnostic_node_recall'](metric,graph,truth)),
                stem=stem,embryo=stem.split('_')[0])
            rows[arm].append(row); print(json.dumps(dict(arm=arm,**row)),flush=True)
    summaries={a:metric.summarise(v) for a,v in rows.items()}
    embryos={a:{e:metric.summarise([r for r in v if r['embryo']==e]) for e in ('44b6','6bba')} for a,v in rows.items()}
    return dict(status='verified_cached_focus_owned_flow_full_score',contract=policy,per_movie=rows,
        summaries=summaries,by_embryo=embryos,comparison=compare(rows,summaries,embryos),
        full_graph_and_motion_replay=True,authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        source_sha256=dict(notebook=hashlib.sha256(notebook.read_bytes()).hexdigest(),
            manifest=hashlib.sha256((folder/'outputs/flow_manifest.json').read_bytes()).hexdigest()),
        validation_scope=policy['scope'],new_target_movies_opened=0,authorized_for_submission=False,
        caveat='Previously exposed complete-movie diagnostic; not fresh independent confirmation or public-best evidence.')


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list): return [finite_json(v) for v in value]
    return value


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists(): raise ValueError('Refuse to overwrite completed full score')
    result=score(ROOT/f'.biohub/cache/kernel-outputs/{RUN}/focus_owned_flow_full',
        ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb',ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train')
    target.write_text(json.dumps(finite_json(result),indent=2,allow_nan=False))
    print(json.dumps(finite_json(result['comparison']),indent=2,allow_nan=False))
