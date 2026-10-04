"""Real fork16 feature/reload/optimizer smoke on both previously blocked cases."""
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
from research.trajectory_event_features_v1 import FEATURES, features
from research.trajectory_event_fast_features_v1 import FeatureContext
from research.trajectory_event_supervision_v1 import for_problem
from research.trajectory_event_fast_training_v1 import prepare, hinge, fit
from research.trajectory_event_fast_case_store_v1 import CaseStore


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def main():
    started = time.monotonic()
    name = 'trajectory-event-fork16-learning-smoke-v1'
    target = ROOT / '.biohub/cache' / name
    receipt = ROOT / 'reports/experiments' / (name + '.json')
    assert not target.exists() and not receipt.exists()
    small_path = ROOT / 'reports/experiments/trajectory-event-fork-coverage-v2.json'
    small = read(small_path)
    assert small['all_known_cases_feasible_at_16'] and not small['expanded_vocabulary_adopted']
    target.mkdir()
    result = dict(status='running', source_only=True, selection_or_validation_opened=False,
        production_candidate_changed=False, authorized_for_submission=False, max_fork_children=16,
        small_feasibility_sha256=sha(small_path), source_sha256=sha(Path(__file__)), records=[],
        helper_sha256={n:sha(ROOT / 'research' / n) for n in (
            'trajectory_event_candidates_v1.py', 'trajectory_event_features_v1.py',
            'trajectory_event_fast_features_v1.py', 'trajectory_event_supervision_v1.py',
            'trajectory_event_fast_training_v1.py', 'trajectory_event_fast_case_store_v1.py')})
    def persist():
        result['seconds'] = time.monotonic() - started
        content = json.dumps(result, indent=2, allow_nan=False) + '\n'
        receipt.write_text(content, encoding='utf-8')
        (target / 'RESULT.json').write_text(content, encoding='utf-8')
    persist()
    try:
        for batch, stem, t in ((4, '44b6_eb2880fc', 73), (5, '6bba_48816121', 17)):
            prefix = 'trajectory-event-source-v1-b' + str(batch)
            folder = ROOT / '.biohub/cache' / (prefix + '-features')
            labels_root = ROOT / '.biohub/cache' / (prefix + '-supervision')
            base = ROOT / '.biohub/cache' / (prefix + '-full-output')
            scope = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(batch)) / 'MOVIES.json'
            plan = read(scope)
            assert any(m['stem'] == stem and m['role'] == 'optimization' for m in plan['movies'])
            frozen, supervision = read(folder / 'RESULT.json'), read(labels_root / 'RESULT.json')
            assert frozen['source_scope_sha256'] == supervision['source_scope_sha256'] == sha(scope)
            assert supervision['feature_receipt_sha256'] == sha(folder / 'RESULT.json')
            backup_path = ROOT / 'reports/experiments' / (prefix + '-full-harvest.json')
            backup = read(backup_path)
            assert backup['status'] == 'verified_backup'
            hashes = {r['path']:r['sha256'] for r in backup['records']}
            pp, gp, fp = [folder / (stem + suffix) for suffix in ('-prediction.json', '-groups.npz', '-features.npz')]
            lp = labels_root / (stem + '-labels.npz')
            ip = base / (stem + '-original/pre-postprocess.json')
            for path, key in ((pp, 'prediction_sha256'), (gp, 'groups_sha256'), (fp, 'features_sha256')):
                assert sha(path) == frozen['per_movie'][stem][key]
            assert sha(lp) == supervision['per_movie'][stem]['labels_sha256']
            assert sha(ip) == hashes[ip.relative_to(base).as_posix()]
            graph, initial = read(pp), read(ip)
            groups, edge, labels = arrays(gp), arrays(fp), arrays(lp)
            assert list(edge['feature_names']) == list(FEATURES[:18])
            case, = list(frames(initial, graph, groups, max_fork_children=16, selected_times={t}))
            reference = features(case, graph, edge['features'])
            fast = FeatureContext(graph, edge['features']).features(case)
            assert np.array_equal(reference, fast), 'Expanded event features differ'
            targets, safe, mapping = for_problem(case, labels['target'], labels['safe'])
            prepared, reason = prepare(case, fast, targets, safe)
            assert prepared is not None and reason == 'prepared'
            path = target / (stem + '-t' + str(t) + '.npz')
            np.savez_compressed(path, options=case['options'], features=fast, targets=targets, safe=safe,
                nparents=case['nparents'], nchildren=case['nchildren'], incumbent=case['incumbent'],
                allowed=prepared['allowed'], margin=prepared['margin'])
            record = dict(stem=stem, batch=batch, t=t, path=path.name, sha256=sha(path),
                n_constraints=prepared['n_constraints'], options=len(case['options']), feature_elements=fast.size,
                exact_reference_features=True, scope_sha256=sha(scope), feature_receipt_sha256=sha(folder / 'RESULT.json'),
                supervision_receipt_sha256=sha(labels_root / 'RESULT.json'), backup_sha256=sha(backup_path), **mapping)
            result['records'].append(record)
            persist()
            print(json.dumps(record), flush=True)
        store = CaseStore(target, result['records'], cache_size=2, max_cache_bytes=256*1024**2)
        model = ROOT / '.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
        assert sha(model) == '508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
        anchor = arrays(model)['anchor']
        penalty = np.r_[np.full(18, .1), np.full(12, .02)]
        before = [hinge(store[i], anchor, time_limit=10)[0] for i in range(len(store))]
        weights, history = fit(store, anchor, penalty, epochs=1, learning_rate=.03, seed=20260914, time_limit=10)
        after = [hinge(store[i], weights, time_limit=10)[0] for i in range(len(store))]
        assert np.isfinite(weights).all() and not np.array_equal(anchor, weights)
        assert len(history) == 1 and history[0]['steps'] == 2
        result.update(status='expanded_vocabulary_learning_functionality_passed',
            exact_saved_case_reconstructions=len(store), anchor_sha256=sha(model), before_hinge=before,
            after_hinge=after, optimizer_history=history, coefficients_not_exported=True,
            model_selected=False, cache_bytes=store.cache_bytes)
        persist()
        print(json.dumps({k:v for k,v in result.items() if k not in ('records', 'helper_sha256')}), flush=True)
    except BaseException as error:
        result.update(status='smoke_failed', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
