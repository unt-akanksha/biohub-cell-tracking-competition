"""Verify CPU registration, freeze anchored flows, then training residual screen."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.registration_flow_anchor import anchor
from research.focus_residual_calibration import TRAIN_STEMS,matched_residuals
RUN='focus-registration-probe-v1'
NB_SHA='31f1a8091e64506ce9f2cef485b508464add2ee07d3c133b983fac48d2bf0e60'
FULL=runpy.run_path(str(ROOT/'scripts/score-focus-owned-flow-full.py'))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    notebook=ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')
    if sha(notebook)!=NB_SHA: raise ValueError('Frozen CPU notebook required')
    nb=json.loads(notebook.read_text())
    source=''.join(nb['cells'][1]['source'])
    for variable,manifest,directory in [('sources','source_hashes.json','repo'),('runtime_sources','runtime_hashes.json','runtime')]:
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==variable)
        bundle=ast.literal_eval(node.value)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in bundle.items()}
        if json.loads((folder/manifest).read_text())!=expected: raise ValueError('Executed hash manifest mismatch')
        if any(sha(folder/directory/k)!=v for k,v in expected.items()): raise ValueError('Executed bytes mismatch')
    terminal=json.loads((folder/'launcher_terminal.json').read_text())
    result=json.loads((folder/'outputs/result.json').read_text())
    if (terminal['status']!='completed' or terminal['run_id']!=RUN or terminal['submission_performed'] is not False
            or not 0<terminal['elapsed_seconds']<=3600 or terminal['declared_budget_seconds']!=3600
            or result['status']!='completed_training_image_registration_probe' or result['gpu_used'] is not False
            or result['ground_truth_opened'] is not False or result['synthetic_sign_check_passed'] is not True
            or result['authorized_for_submission'] is not False
            or result['source_hashes']!=json.loads((folder/'runtime_hashes.json').read_text())):
        raise ValueError('Complete bounded image-only execution required')
    if [r['stem'] for r in result['records']]!=list(TRAIN_STEMS): raise ValueError('Exact two training movies required')
    import numpy as np
    for row in result['records']:
        if len(row['frame_hashes'])!=3 or [(r['source_frame'],r['target_frame']) for r in row['transitions']]!=[(0,1),(1,2)]:
            raise ValueError('Six exact frames and four transitions required')
        for tr in row['transitions']:
            for key in ('forward','reverse'):
                context=tr[key]
                shift=np.asarray(context['global_shift_zyx_um'])
                if shift.shape!=(3,) or not np.isfinite(shift).all() or not 0<=context['reliability']<=1:
                    raise ValueError('Finite physical motion required')
                if not np.array_equal(shift,np.asarray(context['global_shift_zyx_voxel'])*[1.625,.40625,.40625]):
                    raise ValueError('Physical scale mismatch')
    return result,terminal


def main():
    import numpy as np
    import tracksdata as td
    started=time.monotonic()
    folder=ROOT/'.biohub/cache/kernel-outputs'/RUN/'focus_registration_probe'
    destination=ROOT/'reports/experiments'/f'{RUN}-result.json'
    cache=ROOT/'.biohub/cache/focus-registration-anchor-probe-v1'
    if destination.exists() or cache.exists(): raise ValueError('No overwrite or re-selection')
    result,terminal=verify(folder)
    full_folder=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full'
    prepared,_,_=FULL['prepare'](full_folder,ROOT/'kaggle/biohub-focus-owned-flow-full-v1/biohub-focus-owned-flow-full-v1.ipynb')
    cache.mkdir()
    frozen=[]
    for row in result['records']:
        stem=row['stem']; sample=full_folder/'outputs'/stem/'sampled_flow.npz'
        with np.load(sample,allow_pickle=False) as data: coords,flow=data['coords'].copy(),data['backward_um'].copy()
        corrected=flow.astype(float)
        for tr in row['transitions']:
            selected=coords[:,0]==tr['target_frame']
            context=tr['forward']
            corrected[selected]=anchor(flow[selected],context['global_shift_zyx_um'],context['reliability'])
        path=cache/(stem+'.npz')
        np.savez_compressed(path,coords=coords,original=flow,anchored=corrected)
        frozen.append(dict(stem=stem,sha256=sha(path),source_sha256=sha(sample)))
    (cache/'prelabel_manifest.json').write_text(json.dumps(dict(records=frozen,all_predictions_saved_before_gt=True),indent=2))
    metric=FULL['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    reference=json.loads((ROOT/'reports/experiments/focus-owned-flow-full-v1-result.json').read_text())
    rows=[]; pooled={arm:[] for arm in ('original','anchored')}
    for record in frozen:
        stem=record['stem']; path=cache/(stem+'.npz')
        if sha(path)!=record['sha256']: raise ValueError('Prelabel prediction changed')
        graph=td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0]
        truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
        er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        expected=next(r for r in reference['per_movie']['candidate'] if r['stem']==stem)
        if any(getattr(er,k)!=expected[k] for k in er._fields): raise ValueError('Training control score replay failed')
        nodes=graph.node_attrs().sort('node_id')
        with np.load(path,allow_pickle=False) as data:
            coords=data['coords']
            if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords): raise ValueError('Node ordering changed')
            mapping={i:g for i,g in enumerate(nodes[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]) if coords[i,0]<3}
            gt_edges=list(truth.edge_attrs().select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET).iter_rows())
            scores={}
            for arm in pooled:
                values=matched_residuals(coords,data[arm],mapping,gt_edges)
                if not len(values): raise ValueError('No eligible ordinary matched training links')
                squared=np.sum(values**2,axis=1)
                pooled[arm].extend(squared.tolist()); scores[arm]=float(squared.mean())
            rows.append(dict(stem=stem,ordinary_links=len(values),mean_squared_residual_um2=scores,
                delta=scores['anchored']-scores['original']))
    means={k:float(np.mean(v)) for k,v in pooled.items()}
    passed=means['anchored']<means['original'] and all(r['delta']<=0 for r in rows)
    output=dict(run_id=RUN,status='completed_training_registration_residual_screen',registration=result,
        launcher=terminal,per_movie=rows,pooled_mean_squared_residual_um2=means,feasibility_gate_passed=passed,
        prediction_manifest=frozen,notebook_sha256=NB_SHA,elapsed_seconds=time.monotonic()-started,
        source_selection_opened=False,new_target_movies_opened=0,authorized_for_submission=False,
        source_hashes={p:sha(ROOT/p) for p in ['research/registration_flow_anchor.py','scripts/score-focus-registration-probe.py',
            'reports/experiments/focus-registration-anchor-v1-design.md']})
    destination.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(per_movie=rows,pooled=means,passed=passed)),flush=True)


if __name__=='__main__': main()
