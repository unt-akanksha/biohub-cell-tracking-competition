"""Consume only a verified eight-movie cache; never infer/truncate missing data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_adaptation_labels import movie_labels,node_matches
VERIFY=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-cache.py'))
RUN='focus-adaptation-labels-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import tracksdata as td
    started=time.monotonic()
    result_path=ROOT/'reports/experiments'/f'{RUN}-result.json';cache=ROOT/'.biohub/cache'/RUN
    if result_path.exists() or cache.exists():raise ValueError('Do not overwrite a label audit')
    files=['research/focus_adaptation_labels.py','research/focus_predicted_training_labels.py',
           'scripts/audit-focus-adaptation-labels.py','scripts/verify-focus-adaptation-cache.py',
           'reports/experiments/focus-adaptation-labels-v1-design.md']
    frozen={p:sha(ROOT/p) for p in files}
    matcher_path=ROOT/'reports/experiments/focus-adaptation-label-matcher-v1-result.json'
    if sha(matcher_path)!='ca41210bcad92aeaa42c4bde83d2f9f7075d45c302816f0f40d7d3f11d82af78':
        raise ValueError('Exact full-training matcher replay required')
    matcher=json.loads(matcher_path.read_text())
    if (matcher['status']!='passed_full_training_edgeless_matcher_replay' or matcher['all_labels_and_statuses_exact'] is not True
        or any(sha(ROOT/p)!=v for p,v in matcher['source_hashes'].items())):
        raise ValueError('Verified matching implementation changed')
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-cache-v1'
    receipt=VERIFY['verify'](folder)
    recorded=ROOT/'reports/experiments/focus-adaptation-cache-v1-result.json'
    if json.loads(recorded.read_text())!=receipt:raise ValueError('Actual raw cache differs from verified report')
    policy=receipt['contract']
    metric=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    cache.mkdir();records=[]
    for record in receipt['records']:
        if record['role']=='replay':continue
        stem=record['stem'];path=folder/'raw_detections'/(stem+'.npz')
        with np.load(path,allow_pickle=False) as data:coords=data['coords'].copy()
        if record['max_frame_nodes']>2048:raise ValueError('Node guard exceeded; no truncation or labels for partial movie')
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
        mapping=node_matches(coords,truth);keys=td.DEFAULT_ATTR_KEYS
        gt_nodes={r[0]:r[1:] for r in truth.node_attrs().select(keys.NODE_ID,'t','z','y','x').iter_rows()}
        gt_edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
        labels=movie_labels(coords,mapping,gt_nodes,gt_edges,movie_role=record['role'])
        labels.update(stem=stem,raw_coordinates_sha256=record['sha256'],ground_truth_node_count=len(gt_nodes),
            ground_truth_edge_count=len(gt_edges),matching='Patched official7um, no graph predictions used for selection')
        target=cache/(stem+'.json');target.write_text(json.dumps(labels,indent=2,allow_nan=False))
        records.append(dict(stem=stem,role=record['role'],raw_sha256=record['sha256'],labels_sha256=sha(target),
            counts=labels['counts'],windows=len(labels['rows']),nodes=len(coords),max_frame_nodes=record['max_frame_nodes']))
        print(json.dumps(records[-1]),flush=True)
    totals={role:dict(sum((Counter(r['counts']) for r in records if r['role']==role),Counter())) for role in ('fitting','diagnostic')}
    if frozen!={p:sha(ROOT/p) for p in files}:raise ValueError('Label audit source changed during execution')
    report=dict(status='completed_eight_movie_focus_adaptation_label_inventory',run_id=RUN,records=records,totals=totals,
        contract=policy,raw_cache_receipt_sha256=sha(recorded),source_hashes=frozen,
        full_training_matcher_receipt_sha256=sha(matcher_path),
        all_raw_outputs_verified_before_label_access=True,source_selection_opened=False,new_target_movies_opened=0,
        optimizer_run=False,automatic_training_launch=False,authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started,gpu_seconds=0,
        caveat='Four diagnostic movies are excluded from future FOCUS adaptation only, not from original checkpoint training; no independent score established')
    result_path.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(totals=totals)),flush=True)


if __name__=='__main__':main()
