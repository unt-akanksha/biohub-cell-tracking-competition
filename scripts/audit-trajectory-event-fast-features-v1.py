"""Exact replay of all frozen batch-0 event features, without changing models."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_fast_features_v1 import FeatureContext


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();receipt=ROOT/'reports/experiments/trajectory-event-fast-features-v1-audit.json'
    assert not receipt.exists()
    case_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-cases';manifest=read(case_root/'RESULT.json')
    assert manifest['status']=='source_event_cases_prepared' and manifest['training_allowed']
    feature_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features'
    assert sha(feature_root/'RESULT.json')==manifest['feature_receipt_sha256'];frozen=read(feature_root/'RESULT.json')
    base=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-full-output'
    backup=read(ROOT/'reports/experiments/trajectory-event-source-v1-b0-full-harvest.json')
    assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
    results={};total_elements=0
    for stem,movie in manifest['per_movie'].items():
        before=time.monotonic();records={r['t']:r for r in movie['cases'] if r['reason']=='prepared'}
        row=frozen['per_movie'][stem]
        pp=feature_root/(stem+'-prediction.json');gp=feature_root/(stem+'-groups.npz');fp=feature_root/(stem+'-features.npz')
        ip=base/(stem+'-original/pre-postprocess.json')
        assert sha(pp)==row['prediction_sha256'] and sha(gp)==row['groups_sha256'] and sha(fp)==row['features_sha256']
        assert sha(ip)==hashes[ip.relative_to(base).as_posix()]
        graph=read(pp);context=FeatureContext(graph,arrays(fp)['features']);checked=set();feature_seconds=0.
        for case in frames(read(ip),graph,arrays(gp),selected_times=set(records)):
            row=records[case['t']];path=case_root/row['path'];assert sha(path)==row['sha256']
            expected=arrays(path)
            assert np.array_equal(case['options'],expected['options'])
            tick=time.monotonic();actual=context.features(case);feature_seconds+=time.monotonic()-tick
            assert np.array_equal(actual,expected['features']), 'Feature arithmetic drift: '+stem+':'+str(case['t'])
            total_elements+=actual.size;checked.add(case['t'])
        assert checked==set(records)
        results[stem]=dict(exact_frames=len(checked),feature_seconds=feature_seconds,total_seconds=time.monotonic()-before)
        print(json.dumps(dict(stem=stem,**results[stem])),flush=True)
    result=dict(status='exact_source_feature_replay_passed',per_movie=results,elements_compared=total_elements,
        case_manifest_sha256=sha(case_root/'RESULT.json'),source_sha256=sha(Path(__file__)),
        accelerated_features_sha256=sha(ROOT/'research/trajectory_event_fast_features_v1.py'),
        reference_features_sha256=sha(ROOT/'research/trajectory_event_features_v1.py'),
        model_changed=False,ground_truth_opened=False,selection_or_validation_opened=False,
        authorized_for_submission=False,seconds=time.monotonic()-started)
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],elements=total_elements,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
