"""Source-only counterfactual audit of labels ignored by the rejected gate.

Never fits a model or exports an oracle-selected graph. Selection is fixed from
old partial-neutral labels and edge-count changes, before official rescoring.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_correction_gate_v1 import apply_mask


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def assert_baseline_replay(base, previous, scorer):
    expected = scorer.summarise([previous])
    for key in ('score', 'edge_jaccard'):
        assert abs(base['summary'][key] - expected[key]) < 1e-12, key
    for key, value in base['counts'].items():
        assert value == previous[key], key


def neutral_indices(labels, parts, complete_only=False):
    assert len(labels) == len(parts)
    return [i for i, (label, part) in enumerate(zip(labels, parts))
            if label['label'] == -1 and len(part['removed']) != len(part['added'])
            and (not complete_only or (label.get('known_children', 0) > 0
                 and not label.get('unknown_children', 0) and not label.get('ambiguous_children', 0)))]


def main(smoke, attempt, all_known=False):
    started = time.monotonic()
    name = 'trajectory-correction-metric-labels-v2-' + ('known' if all_known else 'smoke' if smoke else 'full')
    if attempt > 1:
        name += '-r' + str(attempt)
    folder = ROOT / '.biohub/cache/trajectory-correction-source-v1'
    dataset = read(folder / 'RESULT.json')
    assert sha(folder / 'RESULT.json') == 'e279e24348d17e60e07b8e6bb964ffab95490e502a3f70ac67ecd35928dfa1db'
    assert dataset['source_only'] and len(dataset['records']) == 60
    assert sha(ROOT / 'research/trajectory_correction_gate_v1.py') == dataset['gate_helper_sha256']
    target = ROOT / '.biohub/cache' / name
    receipt = ROOT / 'reports/experiments' / (name + '.json')
    assert not target.exists() and not receipt.exists(), 'Inspect existing attempt'
    selected = []
    # Prefer known-child neutral cases for the full target-mismatch audit.
    # Neither ordering reads new metric outcomes; smoke retains its first plan.
    for embryo in ('44b6', '6bba'):
        eligible = []
        for stem, item in dataset['records'].items():
            if item['embryo'] != embryo:
                continue
            lp, cp = folder / (stem + '-labels.json'), folder / (stem + '-components.json')
            assert sha(lp) == item['labels_sha256'] and sha(cp) == item['components_sha256']
            labels, parts = read(lp), read(cp)
            indices = neutral_indices(labels, parts, complete_only=all_known)
            if indices:
                if not smoke:
                    indices.sort(key=lambda i: (labels[i].get('known_children', 0) == 0, i))
                no_known = not any(labels[i].get('known_children', 0) > 0 for i in indices)
                eligible.append((False if smoke else no_known,
                                 (ROOT / item['baseline_path']).stat().st_size, stem, indices))
        chosen = sorted(eligible) if all_known else sorted(eligible)[:1 if smoke else 2]
        for _, _, stem, indices in chosen:
            selected.append(dict(stem=stem, indices=indices if all_known else indices[:1 if smoke else 3]))
    if smoke:
        selected = selected[:1]
    assert selected
    if all_known:
        assert sum(len(r['indices']) for r in selected) == 36
    target.mkdir()
    report = dict(status='counterfactual_plan_frozen', source_only=True,
                  selection_or_validation_opened=False, model_fitted=False,
                  authorized_for_submission=False, candidate_changed=False,
                  oracle_graphs_exported=False, source_sha256=sha(Path(__file__)),
                  dataset_sha256=sha(folder / 'RESULT.json'), all_fully_known_neutral_cases=all_known,
                  plan=selected, records={})

    def persist():
        report['seconds'] = time.monotonic() - started
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        (target / 'RESULT.json').write_text(text, encoding='utf-8')
        receipt.write_text(text, encoding='utf-8')

    persist()
    try:
        helper = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
        scorer = helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
        manifest = truth_root / 'train_geff_cache_manifest.json'
        assert sha(manifest) == '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files = read(manifest)['files']
        for choice in selected:
            stem = choice['stem']
            item = dataset['records'][stem]
            scope = read(ROOT / '.biohub/cache/trajectory-event-source-v1-plan' /
                         ('batch-' + str(item['batch'])) / 'MOVIES.json')
            assert any(m['stem'] == stem and m['role'] == 'optimization' for m in scope['movies'])
            bp, cp = ROOT / item['baseline_path'], ROOT / item['candidate_path']
            assert sha(bp) == item['baseline_sha256'] and sha(cp) == item['candidate_sha256']
            baseline, candidate = read(bp), read(cp)
            initial_path = ROOT / item['initial_path']
            backup = read(ROOT / 'reports/experiments' /
                          ('trajectory-event-source-v1-b' + str(item['batch']) + '-full-harvest.json'))
            assert backup['status'] == 'verified_backup'
            initial_record = next(r for r in backup['records']
                                  if r['path'] == stem + '-original/pre-postprocess.json')
            assert sha(initial_path) == initial_record['sha256']
            initial = read(initial_path)
            labels = read(folder / (stem + '-labels.json'))
            parts = read(folder / (stem + '-components.json'))
            matched = [r for r in files if r['relative_path'].startswith(stem + '.geff/')]
            assert len(matched) == 21
            for row in matched:
                assert sha(truth_root / 'train' / row['relative_path']) == row['sha256']
            truth_path = truth_root / 'train' / (stem + '.geff')
            count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])

            def evaluate(payload):
                validate_graph(payload, 100)
                truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
                pred = helper['prediction_graph'](payload)
                result = scorer.evaluate(pred, truth, scale=(1.625, .40625, .40625), max_distance=7.)
                row = scorer.per_sample_metrics(result, count, scorer.node_recall(pred, truth))
                return dict(counts=result._asdict(), summary=helper['finite'](scorer.summarise([row])))

            base = evaluate(baseline)
            prior_name = ('trajectory-event-anchor-source-v1' if item['batch'] == 0
                          else 'trajectory-event-anchor-expanded-source-v1')
            prior_path = ROOT / 'reports/experiments' / (prior_name + '.json')
            assert sha(prior_path) == dataset['source_reports_sha256'][str(int(item['batch'] > 0))]
            prior = read(prior_path)
            rows = prior['per_movie'] if item['batch'] == 0 else prior['per_movie_rows']
            previous = next(r for r in rows['baseline'] if r['stem'] == stem)
            assert_baseline_replay(base, previous, scorer)
            record = dict(baseline=base, baseline_exact_replay=True, components=[])
            report['records'][stem] = record
            persist()
            for index in choice['indices']:
                mask = np.zeros(len(parts), dtype=bool)
                mask[index] = True
                changed = apply_mask(initial, baseline, candidate, mask)
                assert changed['nodes'] == baseline['nodes']
                result = evaluate(changed)
                delta = {k: result['counts'][k] - base['counts'][k] for k in base['counts']}
                record['components'].append(dict(index=index, old_label=labels[index],
                    removed=len(parts[index]['removed']), added=len(parts[index]['added']),
                    result=result, delta_counts=delta,
                    delta_score=result['summary']['score'] - base['summary']['score']))
                persist()
                print(json.dumps(dict(stem=stem, index=index, delta_counts=delta)), flush=True)
        values = [r for movie in report['records'].values() for r in movie['components']]
        report.update(status='source_counterfactual_labels_audited',
            audited_components=len(values),
            partial_neutral_but_official_score_nonzero=sum(abs(r['delta_score']) > 1e-12 for r in values),
            tp_unchanged_fp_changed=sum(r['delta_counts']['edge_tp'] == 0 and r['delta_counts']['edge_fp'] != 0 for r in values),
            no_gate_failure_causality_claim=True)
        persist()
        print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)
    except BaseException as error:
        report.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--all-known', action='store_true')
    parser.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api='blas'):
        main(args.smoke, args.attempt, args.all_known)
