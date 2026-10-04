"""Freeze prediction-only event inputs for the existing ten-movie selection set."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_candidate_ranker_v1 import FEATURES,features


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-event-selection-v1-features'
    receipt=ROOT/'reports/experiments/trajectory-event-selection-v1-features.json'
    assert not target.exists() and not receipt.exists()
    scope=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-plan/MOVIES.json'
    assert sha(scope)=='1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e'
    plan=read(scope);stems=[m['stem'] for m in plan['movies']]
    assert len(stems)==10 and all(m['role']=='selection' for m in plan['movies'])
    source_scope=read(ROOT/'.biohub/cache/trajectory-event-source-v1-plan/RESULT.json')
    # The pinned native role plan is authoritative for disjointness.
    native=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert sha(native)=='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    native_plan=read(native)
    source_stems={m['stem'] for m in native_plan['movies'] if m['role']=='optimization'}
    assert not set(stems)&source_stems
    old_root=ROOT/'.biohub/cache/trajectory-structured-selection-v1'
    old_report=read(old_root/'RESULT.json')
    assert old_report['status']=='independent_selection_diagnostic_complete' and not old_report['selection_used_for_training']
    baseline=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-full-output'
    backup=read(ROOT/'reports/experiments/trajectory-ranker-selection-v1-full-harvest.json')
    assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
    model=read(ROOT/'.biohub/cache/trajectory-structured-portable-v1/structured-trajectory-model.json')
    for name in ('trajectory_candidate_inventory_v1.py','trajectory_candidate_ranker_v1.py'):
        assert sha(ROOT/'research'/name)==model['inference_source_sha256'][name]
    target.mkdir();records={}
    for stem in stems:
        pp=old_root/(stem+'-prediction.json')
        assert sha(pp)==old_report['prediction_sha256'][stem]
        initial_path=baseline/(stem+'-original/pre-postprocess.json')
        raw_path=baseline/(stem+'-original/raw-candidates.npz')
        for path in (initial_path,raw_path):assert sha(path)==hashes[path.relative_to(baseline).as_posix()]
        initial=read(initial_path);graph=read(pp);validate_graph(graph,100)
        groups=candidates(initial,graph)
        with np.load(raw_path,allow_pickle=False) as raw:matrix=features(initial,graph,groups,raw['edges'])
        gp=target/(stem+'-groups.npz');fp=target/(stem+'-features.npz');bp=target/(stem+'-prediction.json')
        np.savez_compressed(gp,**groups);np.savez_compressed(fp,features=matrix,feature_names=np.asarray(FEATURES))
        bp.write_text(json.dumps(graph,sort_keys=True,allow_nan=False),encoding='utf-8')
        records[stem]=dict(groups_sha256=sha(gp),features_sha256=sha(fp),prediction_sha256=sha(bp),
            previous_prediction_sha256=sha(pp),initial_sha256=sha(initial_path),raw_sha256=sha(raw_path),
            queries=len(groups['children']),candidate_edges=len(groups['parents']))
        print(json.dumps(dict(stem=stem,queries=len(groups['children']),edges=len(groups['parents']))),flush=True)
    report=dict(status='selection_event_inputs_frozen_no_new_labels',per_movie=records,
        scope_sha256=sha(scope),native_role_plan_sha256=sha(native),baseline_score_receipt_sha256=sha(old_root/'RESULT.json'),
        source_sha256=sha(Path(__file__)),source_role_disjoint=True,model_fitted=False,
        ground_truth_opened=False,selection_labels_used=False,prior_pipeline_selection_exposure_disclosed=True,
        authorized_for_submission=False,seconds=time.monotonic()-started)
    text=json.dumps(report,indent=2)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
