"""Bounded source-only fork representability and solver pilot, no graph export."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_assignment_v1 import infer


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def main():
    started = time.monotonic()
    destination = ROOT / 'reports/experiments/trajectory-event-problems-v1-audit.json'
    assert not destination.exists()
    plan = ROOT / '.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan) == '5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    stems = [m['stem'] for m in json.loads(plan.read_text())['movies']]
    inventory = json.loads((ROOT / 'reports/experiments/trajectory-candidate-supervision-v1.json').read_text())
    labels = json.loads((ROOT / 'reports/experiments/trajectory-source-label-coverage-v2.json').read_text())
    feature_report = json.loads((ROOT / '.biohub/cache/trajectory-candidate-ranker-v1/RESULT.json').read_text())
    model_path = ROOT / '.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
    assert sha(model_path) == 'e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
    weights = arrays(model_path)['6bba']
    base = ROOT / '.biohub/cache/trajectory-disagreement-source-v1-full-output'
    backup = json.loads((ROOT / 'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert backup['status'] == 'verified_backup'
    for r in backup['records']:
        assert sha(base / r['path']) == r['sha256']
    results, totals = {}, Counter()
    for stem in stems:
        folder = base / (stem + '-original')
        initial = json.loads((folder / 'pre-postprocess.json').read_text())
        final = json.loads((folder / 'repaired-prediction.json').read_text())
        gp = ROOT / '.biohub/cache/trajectory-candidate-supervision-v1' / (stem + '-candidates.npz')
        lp = ROOT / '.biohub/cache/trajectory-source-label-coverage-v2' / (stem + '-physical-labels.npz')
        fp = ROOT / '.biohub/cache/trajectory-candidate-ranker-v1' / (stem + '-features.npz')
        assert sha(gp) == inventory['candidate_sha256'][stem]
        assert sha(lp) == labels['label_sha256'][stem]
        assert sha(fp) == feature_report['features_sha256'][stem]
        groups, target, features = arrays(gp), arrays(lp)['target'], arrays(fp)['features']
        edge_scores = features.astype(np.float64) @ weights
        by_event = defaultdict(list)
        for child, parent in zip(groups['children'], target):
            if parent >= 0:
                by_event[(int(final['nodes'][str(child)]['t']), int(parent))].append(int(child))
        forks = {key: children for key, children in by_event.items() if len(children) == 2}
        event_times = {key[0] for key in forks}
        possible = {int(final['nodes'][str(c)]['t']) for c in groups['children']}
        ordinary = sorted(possible - event_times, key=lambda t: hashlib.sha256((stem + ':' + str(t)).encode()).hexdigest())[:1]
        selected = event_times | set(ordinary)
        records = []
        for case in frames(initial, final, groups, selected_times=selected):
            before = time.monotonic()
            scores = np.zeros(len(case['options']))
            for slot in (0, 1):
                valid = case['edge_indices'][:, slot] >= 0
                scores[valid] += edge_scores[case['edge_indices'][valid, slot]]
            scores[case['options'][:, 2] >= 0] -= 6.
            # The fixed -6 fork cost is solely a runtime-pilot coefficient.
            # No inferred graph is assembled, scored, saved or promoted.
            _, evidence = infer(case, scores, case['incumbent'], time_limit=2.)
            elapsed = time.monotonic() - before
            pindex = {int(p): i for i, p in enumerate(case['parents'])}
            cindex = {int(c): i for i, c in enumerate(case['children'])}
            option_set = {tuple(row) for row in case['options']}
            available, missing = 0, 0
            for (t, parent), children in forks.items():
                if t != case['t']:
                    continue
                if parent not in pindex or any(c not in cindex for c in children):
                    missing += 1
                    continue
                a, b = sorted(cindex[c] for c in children)
                available += int((pindex[parent], a, b) in option_set)
                missing += int((pindex[parent], a, b) not in option_set)
            record = dict(t=case['t'], parents=case['nparents'], children=case['nchildren'],
                          options=len(scores), fork_options=int((case['options'][:, 2] >= 0).sum()),
                          known_forks_representable=available, known_forks_not_representable=missing,
                          solve_seconds=elapsed, solver_fallback=evidence['fallback'])
            records.append(record)
            totals['known_forks_representable'] += available
            totals['known_forks_not_representable'] += missing
            totals['solver_fallbacks'] += int(evidence['fallback'])
        totals['known_matched_forks_in_source_labels'] += len(forks)
        results[stem] = records
        print(json.dumps(dict(stem=stem, cases=records)), flush=True)
    result = dict(status='source_event_solver_pilot_complete', per_movie=results, totals=dict(totals),
                  source_only=True, ground_truth_files_opened=False, existing_source_labels_used=True,
                  selection_or_validation_opened=False, model_fitted=False, graph_predictions_exported=False,
                  quality_gain_established=False, runtime_coefficients_not_a_candidate=True,
                  source_scope_sha256=sha(plan), source_sha256=sha(Path(__file__)),
                  candidate_module_sha256=sha(ROOT / 'research/trajectory_event_candidates_v1.py'),
                  solver_module_sha256=sha(ROOT / 'research/trajectory_event_assignment_v1.py'),
                  elapsed_seconds=time.monotonic() - started)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], totals=result['totals'], seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
