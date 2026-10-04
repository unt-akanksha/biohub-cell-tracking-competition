"""Prove edgeless matching agrees with previous full-training scored labels."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_adaptation_labels import node_matches
from research.focus_predicted_training_labels import targets


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import tracksdata as td
    result_path=ROOT/'reports/experiments/focus-adaptation-label-matcher-v1-result.json'
    if result_path.exists():raise ValueError('No overwrite of completed matching proof')
    receipt_path=ROOT/'reports/experiments/focus-predicted-node-labels-v1-result.json'
    if sha(receipt_path)!='1ea573dfaa1e7075d4c2c486bcca8c7bd27e592ce233dc4afdcc844d8dedbbd1':raise ValueError('Frozen original training label audit required')
    receipt=json.loads(receipt_path.read_text());records=[]
    # Load the pinned scorer for its hash verification, even though edgeless
    # matching directly calls its underlying DistanceMatching implementation.
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    for record in receipt['records']:
        stem=record['stem'];label_path=ROOT/'.biohub/cache/focus-predicted-node-labels-v1'/(stem+'.json')
        sample=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full/outputs'/stem/'sampled_flow.npz'
        if sha(sample)!=record['motion_sha256'] or sha(label_path)!=record['label_sha256']:raise ValueError('Exact completed input/label cache required')
        with np.load(sample,allow_pickle=False) as data:coords=data['coords'].copy()
        truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
        mapping=node_matches(coords,truth);keys=td.DEFAULT_ATTR_KEYS
        nodes={r[0]:r[1:] for r in truth.node_attrs().select(keys.NODE_ID,'t','z','y','x').iter_rows()}
        edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
        original=json.loads(label_path.read_text());windows=0
        for row in original['rows']:
            now=targets(coords,mapping,nodes,edges,row['source_frame'])
            converted={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in now.items()}
            if converted!=row:raise ValueError('Edgeless matcher differs from previous fully scored graph labels')
            windows+=1
        records.append(dict(stem=stem,windows_replayed=windows,nodes=len(coords),label_sha256=record['label_sha256']))
    result=dict(status='passed_full_training_edgeless_matcher_replay',records=records,all_labels_and_statuses_exact=True,
        prior_label_receipt_sha256=sha(receipt_path),new_source_or_target_data_opened=False,gpu_used=False,
        authorized_for_submission=False,source_hashes={p:sha(ROOT/p) for p in ['research/focus_adaptation_labels.py',
            'research/focus_predicted_training_labels.py','scripts/verify-focus-adaptation-label-matcher.py']})
    result_path.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
