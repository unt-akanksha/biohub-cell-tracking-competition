"""Audit uniform fork16 feasibility across all 60 optimization-source movies.

This is a candidate-coverage check, not training or score measurement. Ground
truth only supplies partial constraints; it never inserts candidate options.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_supervision_v1 import for_problem
from research.trajectory_event_fast_training_v1 import prepare


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def main():
    reports = ROOT / 'reports/experiments'
    receipt = reports / 'trajectory-event-source-vocabulary16-v1.json'
    assert not receipt.exists(), 'Observe the previous audit; do not duplicate it'
    small_path = reports / 'trajectory-event-fork-coverage-v2.json'
    small = read(small_path)
    assert small['status'] == 'known_source_incompatibilities_audited' and small['all_known_cases_feasible_at_16']
    assert len(small['records']) == 4 and not small['expanded_vocabulary_adopted']
    started = time.monotonic()
    result = dict(status='running', max_fork_children=16, source_only=True,
        small_real_feasibility_test_sha256=sha(small_path), source_sha256=sha(Path(__file__)),
        selection_or_validation_opened=False, model_fitted=False, production_candidate_changed=False,
        source_label_insertion_into_candidate_options=False, per_movie={}, batch_receipts={},
        helper_sha256={name:sha(ROOT / 'research' / name) for name in (
            'trajectory_event_candidates_v1.py', 'trajectory_event_supervision_v1.py',
            'trajectory_event_fast_training_v1.py', 'trajectory_event_assignment_v1.py')})
    totals = Counter()
    def persist():
        result.update(totals=dict(totals), seconds=time.monotonic()-started)
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    persist()
    try:
        for batch in (4, 5, 0, 1, 2, 3, 6, 7):
            prefix = 'trajectory-event-source-v1-b' + str(batch)
            scope = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(batch)) / 'MOVIES.json'
            plan = read(scope)
            assert all(m['role'] == 'optimization' for m in plan['movies'])
            feature_root = ROOT / '.biohub/cache' / (prefix + '-features')
            labels_root = ROOT / '.biohub/cache' / (prefix + '-supervision')
            base = ROOT / '.biohub/cache' / (prefix + '-full-output')
            frozen, supervision = read(feature_root / 'RESULT.json'), read(labels_root / 'RESULT.json')
            assert frozen['status'] == 'current_candidate_source_features_frozen'
            assert supervision['status'] == 'source_event_supervision_prepared' and supervision['source_only']
            assert supervision['source_scope_sha256'] == frozen['source_scope_sha256'] == sha(scope)
            assert supervision['feature_receipt_sha256'] == sha(feature_root / 'RESULT.json')
            backup_path = reports / (prefix + '-full-harvest.json')
            backup = read(backup_path)
            assert backup['status'] == 'verified_backup'
            hashes = {r['path']:r['sha256'] for r in backup['records']}
            result['batch_receipts'][str(batch)] = dict(scope=sha(scope), features=sha(feature_root / 'RESULT.json'),
                supervision=sha(labels_root / 'RESULT.json'), backup=sha(backup_path))
            for movie in plan['movies']:
                stem = movie['stem']
                assert stem not in result['per_movie']
                graph_path = feature_root / (stem + '-prediction.json')
                groups_path = feature_root / (stem + '-groups.npz')
                label_path = labels_root / (stem + '-labels.npz')
                initial_path = base / (stem + '-original/pre-postprocess.json')
                assert sha(graph_path) == frozen['per_movie'][stem]['prediction_sha256']
                assert sha(groups_path) == frozen['per_movie'][stem]['groups_sha256']
                assert sha(label_path) == supervision['per_movie'][stem]['labels_sha256']
                assert sha(initial_path) == hashes[initial_path.relative_to(base).as_posix()]
                initial, graph = read(initial_path), read(graph_path)
                groups, labels = arrays(groups_path), arrays(label_path)
                selected = {int(graph['nodes'][str(c)]['t']) for c,p in zip(groups['children'], labels['target']) if p >= 0}
                rows, counts = [], Counter()
                for case in frames(initial, graph, groups, max_fork_children=16, selected_times=selected):
                    targets, safe, mapping = for_problem(case, labels['target'], labels['safe'])
                    if not np.any(targets >= 0):
                        reason, feasible = 'no_known_parent_constraints', None
                    else:
                        prepared, reason = prepare(case, np.zeros((len(case['options']), 1)), targets, safe)
                        feasible = prepared is not None
                    rows.append(dict(t=case['t'], options=len(case['options']), reason=reason, feasible=feasible, **mapping))
                    counts[reason] += 1
                missing = sorted(selected - {r['t'] for r in rows})
                counts['known_frames_outside_editable_scope'] = len(missing)
                result['per_movie'][stem] = dict(batch=batch, rows=rows, counts=dict(counts), missing_editable_times=missing)
                totals.update(counts)
                persist()
                print(json.dumps(dict(stem=stem, batch=batch, **dict(counts))), flush=True)
        assert len(result['per_movie']) == 60
        result.update(status='full_source_vocabulary_audited',
            all_partial_constraints_compatible=totals['incompatible_partial_constraints'] == 0,
            expanded_vocabulary_adopted=False, training_feature_dataset_prepared=False)
        persist()
    except BaseException as error:
        result.update(status='audit_failed', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
