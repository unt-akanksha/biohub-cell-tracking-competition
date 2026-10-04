"""Prepare separate fork16 training cases only after real smoke and full coverage.

Reuse the pinned source-only extractor; never overwrite the original fork8 data.
"""
import hashlib
import json
from pathlib import Path
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    base = ROOT / 'scripts/prepare-trajectory-event-cases-v1.py'
    assert sha(base) == '259359d59978a8a254cd3ec98ca35786ff157dbe2ff47345abe1aface25be35e'
    coverage_path = ROOT / 'reports/experiments/trajectory-event-source-vocabulary16-v1.json'
    coverage = read(coverage_path)
    assert coverage['status'] == 'full_source_vocabulary_audited'
    assert coverage['all_partial_constraints_compatible'] and len(coverage['per_movie']) == 60
    assert coverage['totals']['known_frames_outside_editable_scope'] == 0
    smoke_path = ROOT / 'reports/experiments/trajectory-event-fork16-learning-smoke-v1.json'
    smoke = read(smoke_path)
    assert smoke['status'] == 'expanded_vocabulary_learning_functionality_passed'
    assert len(smoke['records']) == smoke['exact_saved_case_reconstructions'] == 2
    assert smoke['coefficients_not_exported'] and not smoke['model_selected']
    assert smoke['optimizer_history'][0]['steps'] == 2
    for manifest in (coverage, smoke):
        assert manifest['source_only'] and not manifest['selection_or_validation_opened']
        assert not manifest['production_candidate_changed']
        for name, expected in manifest['helper_sha256'].items():
            assert sha(ROOT / 'research' / name) == expected, 'Helper changed: ' + name
    source = base.read_text(encoding='utf-8')
    changes = [
        ("target=ROOT/'.biohub/cache'/(name+'-cases')", "target=ROOT/'.biohub/cache'/(name+'-cases-fork16')"),
        ("receipt=ROOT/'reports/experiments'/(name+'-cases.json')", "receipt=ROOT/'reports/experiments'/(name+'-cases-fork16.json')"),
        ('from research.trajectory_event_features_v1 import FEATURES, features',
         'from research.trajectory_event_features_v1 import FEATURES\nfrom research.trajectory_event_fast_features_v1 import FeatureContext'),
        ('from research.trajectory_event_assignment_v1 import prepare',
         'from research.trajectory_event_fast_training_v1 import prepare'),
        ('graph=read(pp);initial=read(ip);groups=arrays(gp);labels=arrays(lp);edge=arrays(fp)',
         "graph=read(pp);initial=read(ip);groups=arrays(gp);labels=arrays(lp);edge=arrays(fp)\n            context=FeatureContext(graph,edge['features'])"),
        ('frames(initial,graph,groups,selected_times=selected)',
         'frames(initial,graph,groups,max_fork_children=16,selected_times=selected)'),
        ("x=features(problem,graph,edge['features'])", 'x=context.features(problem)'),
        ("source_sha256=sha(Path(__file__)),feature_names=list(FEATURES),",
         "source_sha256=sha(Path(__file__)),feature_names=list(FEATURES),max_fork_children=16,\n"
         "                full_source_coverage_sha256=" + repr(sha(coverage_path)) + ",\n"
         "                real_learning_smoke_sha256=" + repr(sha(smoke_path)) + ","),
        ("'trajectory_event_features_v1.py','trajectory_event_supervision_v1.py','trajectory_event_assignment_v1.py')",
         "'trajectory_event_features_v1.py','trajectory_event_fast_features_v1.py','trajectory_event_supervision_v1.py',\n"
         "                    'trajectory_event_assignment_v1.py','trajectory_event_fast_training_v1.py')"),
        ("        report['status']='source_event_cases_prepared'",
         "        coverage=read(ROOT/'reports/experiments/trajectory-event-source-vocabulary16-v1.json')\n"
         "        for stem,item in report['per_movie'].items():\n"
         "            assert {(r['t'],r['reason']) for r in item['cases']} == {(r['t'],r['reason']) for r in coverage['per_movie'][stem]['rows']}\n"
         "        report['status']='source_event_cases_prepared'"),
    ]
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=__file__))


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
