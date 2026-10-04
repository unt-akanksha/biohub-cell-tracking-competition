"""Verify exact saved owned-head predictions before fresh training-only scoring."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_owned_neural_links import link
from research.backward_flow_linking import link_backward_flow
RUN='focus-owned-neural-probe-v1'
NB_SHA='9f52f7ba5a46b9ea5f75a36668d86e0453fc4119781448d29e759a36e7e7fe11'
BASE=runpy.run_path(str(ROOT/'scripts/score-focus-division-preserving-assignment.py'))
SCORER=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(rows):
    import math
    if [r['stem'] for r in rows['control']]!=['6bba_f1fde7e0','6bba_23af9eeb'] or [r['stem'] for r in rows['candidate']]!=[r['stem'] for r in rows['control']]:
        raise ValueError('Exact training pair order required')
    scores={}
    for arm,values in rows.items():
        counts=[sum(r[k] for r in values) for k in ('edge_tp','edge_fp','edge_fn')]
        if not all(math.isfinite(c) and c>=0 for c in counts) or sum(counts)==0: raise ValueError('Finite nonempty counts required')
        scores[arm]=counts[0]/sum(counts)
    conditions=dict(raw_edge_jaccard_gain=scores['candidate']>scores['control'],
        same_node_counts=all(c['num_pred_nodes']==b['num_pred_nodes'] for c,b in zip(rows['candidate'],rows['control'])),
        every_movie_correct_edges_preserved=all(c['edge_tp']>=b['edge_tp'] for c,b in zip(rows['candidate'],rows['control'])),
        every_movie_ordinary_correct_edges_preserved=all(c['ordinary_tp']>=b['ordinary_tp'] for c,b in zip(rows['candidate'],rows['control'])),
        every_movie_true_divisions_preserved=all(c['division_tp']>=b['division_tp'] for c,b in zip(rows['candidate'],rows['control'])))
    return dict(pooled_raw_edge_jaccard=scores,conditions=conditions,feasibility_gate_passed=all(conditions.values()))


def main():
    import numpy as np
    import tracksdata as td
    folder=ROOT/'.biohub/cache/kernel-outputs'/RUN/'focus_owned_neural_probe'
    result_path=ROOT/'reports/experiments'/f'{RUN}-result.json'
    if result_path.exists(): raise ValueError('Do not overwrite completed result')
    notebook=ROOT/'kaggle'/('biohub-'+RUN)/('biohub-'+RUN+'.ipynb')
    if sha(notebook)!=NB_SHA:raise ValueError('Exact frozen notebook required')
    source=''.join(json.loads(notebook.read_text())['cells'][1]['source']);bundles={}
    for variable,manifest,directory in [('sources','source_hashes.json','repo'),('runtime_sources','runtime_hashes.json','runtime')]:
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==variable)
        bundle=ast.literal_eval(node.value);bundles[variable]=bundle
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in bundle.items()}
        if json.loads((folder/manifest).read_text())!=expected or any(sha(folder/directory/k)!=v for k,v in expected.items()):raise ValueError('Actual executed source bytes changed')
    terminal=json.loads((folder/'launcher_terminal.json').read_text());result=json.loads((folder/'outputs/result.json').read_text())
    if (terminal['status']!='completed' or terminal['run_id']!=RUN or terminal['submission_performed'] is not False
        or not 0<terminal['elapsed_seconds']<=3600 or result['status']!='completed_focus_owned_neural_probe'
        or result['model_before']!=result['model_after'] or result['ground_truth_opened'] is not False
        or result['authorized_for_submission'] is not False
        or result['checkpoint_sha256']!='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'):
        raise ValueError('Complete frozen owned-checkpoint execution required')
    inputs=json.loads(bundles['runtime_sources']['training_inputs.json']);prepared={}
    if [r['stem'] for r in result['records']]!=[r['stem'] for r in inputs]:raise ValueError('Wrong training scope')
    for record,original in zip(result['records'],inputs):
        stem=record['stem'];path=folder/'outputs'/(stem+'.npz')
        if sha(path)!=record['sha256'] or record['input_sha256']!=original['source_sha256']:raise ValueError('Prediction/input identity mismatch')
        sample=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full/outputs'/stem/'sampled_flow.npz'
        if sha(sample)!=original['source_sha256']:raise ValueError('Original full motion input changed')
        with np.load(path,allow_pickle=False) as data, np.load(sample,allow_pickle=False) as old:
            coords=data['coords'].copy();flow=data['backward_um'].copy();selected=old['coords'][:,0]<3
            if not np.array_equal(coords,old['coords'][selected]) or not np.array_equal(flow,old['backward_um'][selected]):raise ValueError('Raw nodes or motion changed')
            scores={i:data[f'neural_{i}'].copy() for i in range(2)}
            for arm,matrix in [('control',{i:np.zeros_like(v) for i,v in scores.items()}),('candidate',scores)]:
                expected=np.asarray(link(coords,flow,matrix),dtype=float).reshape(-1,3)
                actual=data[arm+'_edges'].copy()
                if (actual.shape!=expected.shape or not np.array_equal(actual[:,:2],expected[:,:2])
                    or not np.allclose(actual[:,2],expected[:,2],rtol=1e-12,atol=1e-14)):
                    raise ValueError('Host edge identity/probability replay failed')
                if arm=='control' and [(int(s),int(d)) for s,d,p in actual]!=[(s,d) for s,d,p in link_backward_flow(coords,flow)]:raise ValueError('Original physical-flow graph differs')
                prepared[(stem,arm)]=(coords,actual[:,:2].astype(np.int64))
    # Only now open training annotations: every prediction and control verified.
    metric=SCORER['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot');rows={'control':[],'candidate':[]}
    for stem in [r['stem'] for r in inputs]:
        for arm in rows:
            truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
            truth=truth.filter(td.NodeAttr('t')<3).subgraph()
            graph=BASE['graph_from_arrays'](*prepared[(stem,arm)])
            er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            keys=td.DEFAULT_ATTR_KEYS
            truth_edges=set(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows());degree=Counter(s for s,d in truth_edges)
            mapping=dict(graph.node_attrs().select(keys.NODE_ID,keys.MATCHED_NODE_ID).iter_rows())
            correct={(mapping[s],mapping[d]) for s,d in graph.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows()
                if (mapping[s],mapping[d]) in truth_edges}
            if len(correct)!=er.edge_tp:raise ValueError('Ordinary-link decomposition inconsistent with scorer')
            rows[arm].append(dict(stem=stem,**er._asdict(),ordinary_tp=sum(degree[s]==1 for s,d in correct)))
    output=dict(status='completed_verified_training_neural_probe',run_id=RUN,per_movie=rows,comparison=gate(rows),
        gpu_receipt=result,launcher=terminal,notebook_sha256=NB_SHA,all_predictions_verified_before_gt=True,
        new_target_movies_opened=0,source_selection_opened=False,authorized_for_submission=False,
        source_hashes={p:sha(ROOT/p) for p in ['research/focus_owned_neural_links.py','scripts/score-focus-owned-neural-probe.py','reports/experiments/focus-owned-neural-probe-v1-design.md']})
    result_path.write_text(json.dumps(output,indent=2,allow_nan=False));print(json.dumps(dict(per_movie=rows,comparison=output['comparison'])),flush=True)


if __name__=='__main__':main()
