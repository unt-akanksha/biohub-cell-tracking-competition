"""CPU-only full training-cache replay and conservative sparse label inventory."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_predicted_training_labels import targets,role
from research.focus_residual_calibration import TRAIN_STEMS
FULL=runpy.run_path(str(ROOT/'scripts/score-focus-owned-flow-full.py'))
RUN='focus-predicted-node-labels-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import tracksdata as td
    started=time.monotonic()
    cache=ROOT/'.biohub/cache'/RUN;report=ROOT/'reports/experiments'/f'{RUN}-result.json'
    if cache.exists() or report.exists():raise ValueError('Never overwrite generated labels or audit')
    files=['research/focus_predicted_training_labels.py','scripts/audit-focus-predicted-training-labels.py',
           'reports/experiments/focus-predicted-node-labels-v1-design.md']
    frozen={p:sha(ROOT/p) for p in files}
    reference_path=ROOT/'reports/experiments/focus-owned-flow-full-v1-result.json'
    if sha(reference_path)!='079abeeccec5f7a76e27b40c88063fae30bca4a155f3dae3bf3bc0311e4f6685':raise ValueError('Frozen full training result required')
    reference=json.loads(reference_path.read_text())
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full'
    prepared,_,_=FULL['prepare'](folder,ROOT/'kaggle/biohub-focus-owned-flow-full-v1/biohub-focus-owned-flow-full-v1.ipynb')
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]
    if not set(TRAIN_STEMS)<=set(split['train']) or set(TRAIN_STEMS)&set(split['selection']+split['audit_order']):raise ValueError('Original training movies only')
    metric=FULL['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    cache.mkdir();records=[]
    for stem in TRAIN_STEMS:
        sample=folder/'outputs'/stem/'sampled_flow.npz'
        with np.load(sample,allow_pickle=False) as data:coords=data['coords'].copy()
        graph=td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0]
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
        er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        expected=next(r for r in reference['per_movie']['candidate'] if r['stem']==stem)
        if any(getattr(er,k)!=expected[k] for k in er._fields):raise ValueError('Complete training control score replay failed')
        keys=td.DEFAULT_ATTR_KEYS;nodes=graph.node_attrs().sort(keys.NODE_ID)
        if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords):raise ValueError('Raw node order changed')
        matches={i:g for i,g in enumerate(nodes[keys.MATCHED_NODE_ID])}
        truth_nodes={r[0]:r[1:] for r in truth.node_attrs().select(keys.NODE_ID,'t','z','y','x').iter_rows()}
        truth_edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
        rows=[];counts={r:Counter() for r in ('fitting','diagnostic','embargo')}
        for t in range(99):
            result=targets(coords,matches,truth_nodes,truth_edges,t)
            counts[result['role']].update(result['status'])
            if result['role']=='embargo':continue
            rows.append({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()})
        payload=dict(stem=stem,input_motion_sha256=sha(sample),rows=rows,counts={r:dict(v) for r,v in counts.items()},
            matching='Patched official7um, unique matched nodes; ambiguous/unknown targets ignored',
            training_frames=list(range(70)),diagnostic_frames=list(range(80,100)),labels_from_original_training_only=True)
        path=cache/(stem+'.json');path.write_text(json.dumps(payload,indent=2,allow_nan=False))
        records.append(dict(stem=stem,motion_sha256=sha(sample),label_sha256=sha(path),
            nodes=len(coords),windows=len(rows),counts=payload['counts'],control_counts_replayed=True))
        print(json.dumps(records[-1]),flush=True)
    totals={r:dict(sum((Counter(row['counts'][r]) for row in records),Counter())) for r in ('fitting','diagnostic','embargo')}
    output=dict(status='completed_focus_specific_training_label_inventory',run_id=RUN,records=records,totals=totals,
        source_hashes=frozen,elapsed_seconds=time.monotonic()-started,gpu_seconds=0,
        predictions_changed=False,optimizer_run=False,source_selection_opened=False,new_target_movies_opened=0,
        authorized_for_submission=False,automatic_training_launch=False,
        caveat='Original model already trained on native predicted detections. This cache changes proposal domain to FOCUS; temporal diagnostics are not independent model validation.')
    if frozen!={p:sha(ROOT/p) for p in files}:raise ValueError('Audit source changed')
    report.write_text(json.dumps(output,indent=2,allow_nan=False));print(json.dumps(dict(totals=totals)),flush=True)


if __name__=='__main__':main()
