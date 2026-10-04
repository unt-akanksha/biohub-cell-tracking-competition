"""Inventory additional training labels only after complete raw-cache verification."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_adaptation_labels import movie_labels,node_matches
RUN='focus-extra-fit-labels-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import tracksdata as td
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-result.json';cache=ROOT/'.biohub/cache'/RUN
    if target.exists() or cache.exists():raise ValueError('Never overwrite label inventory')
    sources={p:sha(ROOT/p) for p in ('research/focus_adaptation_labels.py','research/focus_predicted_training_labels.py',
        'research/focus_extra_fit_scope.py','scripts/verify-focus-extra-fit-cache.py','scripts/audit-focus-extra-fit-labels.py',
        'reports/experiments/focus-extra-fit-labels-v1-design.md')}
    original_path=ROOT/'reports/experiments/focus-adaptation-labels-v1-result.json';original=json.loads(original_path.read_text())
    if any(sha(ROOT/p)!=v for p,v in original['source_hashes'].items()):raise ValueError('Original audited sparse-label implementation changed')
    matcher=ROOT/'reports/experiments/focus-adaptation-label-matcher-v1-result.json'
    if sha(matcher)!='ca41210bcad92aeaa42c4bde83d2f9f7075d45c302816f0f40d7d3f11d82af78':raise ValueError('Exact full-training matching proof required')
    raw_folder=ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-cache-v1'
    receipt=runpy.run_path(str(ROOT/'scripts/verify-focus-extra-fit-cache.py'))['verify'](raw_folder)
    receipt_path=ROOT/'reports/experiments/focus-extra-fit-cache-v1-result.json'
    if json.loads(receipt_path.read_text())!=receipt:raise ValueError('Actual complete detector cache must match verified receipt')
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    cache.mkdir();records=[]
    for record in receipt['records']:
        if record['role']=='replay':continue
        stem=record['stem']
        if record['role']!='fitting' or stem in receipt['contract']['unchanged_diagnostic_stems'] or record['max_frame_nodes']>2048:
            raise ValueError('Bounded new fitting movie only; no diagnostic reassignment or truncation')
        with np.load(raw_folder/'raw_detections'/(stem+'.npz'),allow_pickle=False) as saved:coords=saved['coords'].copy()
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0];mapping=node_matches(coords,truth);keys=td.DEFAULT_ATTR_KEYS
        nodes={r[0]:r[1:] for r in truth.node_attrs().select(keys.NODE_ID,'t','z','y','x').iter_rows()}
        edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
        labels=movie_labels(coords,mapping,nodes,edges,movie_role='fitting')
        labels.update(stem=stem,raw_coordinates_sha256=record['sha256'],ground_truth_node_count=len(nodes),ground_truth_edge_count=len(edges))
        path=cache/(stem+'.json');path.write_text(json.dumps(labels,indent=2,allow_nan=False))
        records.append(dict(stem=stem,role='fitting',raw_sha256=record['sha256'],labels_sha256=sha(path),
            counts=labels['counts'],windows=len(labels['rows']),nodes=len(coords),max_frame_nodes=record['max_frame_nodes']))
        print(json.dumps(records[-1]),flush=True)
    totals=dict(sum((Counter(r['counts']) for r in records),Counter()))
    combined=dict(Counter(original['totals']['fitting'])+Counter(totals))
    if any(sha(ROOT/p)!=v for p,v in sources.items()):raise ValueError('Audited label semantics changed during inventory')
    result=dict(status='completed_extra_fitting_label_inventory',run_id=RUN,records=records,new_fitting_totals=totals,
        combined_twelve_movie_fitting_totals=combined,unchanged_diagnostic_totals=original['totals']['diagnostic'],
        raw_receipt_sha256=sha(receipt_path),original_label_receipt_sha256=sha(original_path),source_hashes=sources,
        contract=receipt['contract'],all_raw_outputs_verified_before_label_access=True,optimizer_run=False,
        source_selection_opened=False,new_target_movies_opened=0,authorized_for_submission=False,
        gpu_seconds=0,elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(new_fitting_totals=totals,combined_fitting_totals=combined)))


if __name__=='__main__':main()
