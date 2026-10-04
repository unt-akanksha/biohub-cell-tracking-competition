"""Real movie replay for accelerated features; no labels or model changes."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_event_fast_inference_v1 import refine


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def by_frame(graph):
    result={t:set() for t in range(100)}
    for edge in graph['edges']:
        a,b=int(edge['source_id']),int(edge['target_id'])
        result[int(graph['nodes'][str(b)]['t'])].add((a,b))
    return result


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-event-fast-runtime-v1'
    receipt=ROOT/'reports/experiments/trajectory-event-fast-runtime-v1.json'
    assert not target.exists() and not receipt.exists()
    reference_root=ROOT/'.biohub/cache/trajectory-event-smoke-source-eight-v1';reference=read(reference_root/'RESULT.json')
    assert reference['status']=='full_source_smoke_diagnostic_complete'
    feature_audit=ROOT/'reports/experiments/trajectory-event-fast-features-v1-audit.json';proof=read(feature_audit)
    assert proof['status']=='exact_source_feature_replay_passed'
    assert sha(ROOT/'research/trajectory_event_fast_features_v1.py')==proof['accelerated_features_sha256']
    fit_root=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1';model=fit_root/'epoch-3.npz'
    assert sha(model)==reference['model_sha256']=='508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
    weights=arrays(model)['weights']
    feature_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features';frozen=read(feature_root/'RESULT.json')
    base=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-full-output'
    backup=read(ROOT/'reports/experiments/trajectory-event-source-v1-b0-full-harvest.json')
    assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
    target.mkdir();results={}
    report=dict(status='running',source_only=True,ground_truth_opened=False,model_changed=False,
                authorized_for_submission=False,source_sha256=sha(Path(__file__)),
                feature_audit_sha256=sha(feature_audit),reference_receipt_sha256=sha(reference_root/'RESULT.json'),
                fast_inference_sha256=sha(ROOT/'research/trajectory_event_fast_inference_v1.py'),model_sha256=sha(model))
    def persist():
        report.update(per_movie=results,seconds=time.monotonic()-started)
        text=json.dumps(report,indent=2)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        for stem in ('6bba_6ca87370','6bba_2819ca14','6bba_3abfe10a'):
            row=frozen['per_movie'][stem]
            pp=feature_root/(stem+'-prediction.json');gp=feature_root/(stem+'-groups.npz');fp=feature_root/(stem+'-features.npz')
            ip=base/(stem+'-original/pre-postprocess.json');rp=reference_root/(stem+'-prediction.json')
            assert sha(pp)==row['prediction_sha256'] and sha(gp)==row['groups_sha256'] and sha(fp)==row['features_sha256']
            assert sha(ip)==hashes[ip.relative_to(base).as_posix()] and sha(rp)==reference['inference'][stem]['prediction_sha256']
            graph=read(pp)
            actual,details=refine(read(ip),graph,arrays(gp),arrays(fp)['features'],weights)
            validate_graph(actual,100);assert actual['nodes']==graph['nodes']
            path=target/(stem+'-prediction.json');path.write_text(json.dumps(actual,sort_keys=True,allow_nan=False),encoding='utf-8')
            expected=read(rp);old_frames=by_frame(expected);new_frames=by_frame(actual)
            old_rows={r['t']:r for r in reference['inference'][stem]['frames']}
            new_rows={r['t']:r for r in details['frames']}
            eligible={t for t in old_rows.keys()&new_rows.keys() if not old_rows[t]['fallback'] and not new_rows[t]['fallback']}
            differing=[t for t in sorted(eligible) if old_frames[t]!=new_frames[t]]
            results[stem]=dict(details,reference_seconds=reference['inference'][stem]['seconds'],
                matched_reference_frames=len(eligible)-len(differing),differing_reference_frames=differing,
                reference_coverage=reference['inference'][stem]['processed_frames'],
                prediction_sha256=sha(path),whole_graph_exact=actual==expected)
            persist();print(json.dumps(dict(stem=stem,**{k:v for k,v in results[stem].items() if k!='frames'})),flush=True)
        report.update(status='fast_full_movie_replay_complete',
            exact_on_shared_nonfallback_frames=all(not r['differing_reference_frames'] for r in results.values()),
            all_transitions_completed_without_fallback=all(r['processed_frames']==99 and not r['budget_exhausted'] and not r['solver_fallbacks'] for r in results.values()))
        persist()
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':main()
