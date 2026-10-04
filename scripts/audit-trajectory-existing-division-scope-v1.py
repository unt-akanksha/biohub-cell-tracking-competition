"""Source-only coverage audit; never exports oracle graphs or per-fork GT labels."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import runpy
import sys
import time
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_existing_division_scope_v1 import inventory


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mapped_graph(payload):
    import polars as pl
    import tracksdata as td
    graph = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.)
    graph.add_node_attr_key('audit_source_id', pl.Int64, -1)
    ids = sorted(map(int, payload['nodes']))
    mapped = graph.bulk_add_nodes([dict({k: payload['nodes'][str(i)][k] for k in ('t', 'z', 'y', 'x')}, audit_source_id=i) for i in ids])
    mapping = dict(zip(ids, mapped))
    assert len(mapping) == len(set(mapped)) == len(ids)
    graph.bulk_add_edges([dict(source_id=mapping[e['source_id']], target_id=mapping[e['target_id']]) for e in payload['edges']])
    for row in graph.node_attrs().iter_rows(named=True):
        assert mapping[row['audit_source_id']] == row['node_id']
    return graph, mapping


def main(smoke):
    started = time.monotonic()
    name = 'trajectory-existing-division-scope-v1-' + ('smoke' if smoke else 'full')
    reports = ROOT / 'reports/experiments'
    receipt, target = reports / (name + '.json'), ROOT / '.biohub/cache' / name
    assert not receipt.exists() and not target.exists(), 'Preserve prior audit attempt'
    dataset_path = ROOT / '.biohub/cache/trajectory-correction-source-v1/RESULT.json'
    assert sha(dataset_path) == 'e279e24348d17e60e07b8e6bb964ffab95490e502a3f70ac67ecd35928dfa1db'
    dataset = read(dataset_path)
    assert dataset['source_only'] and len(dataset['records']) == 60
    previous_path = reports / 'trajectory-correction-gate-oof-v1.json'
    assert sha(previous_path) == 'bf735591bfb02bc707d7ec85fc8dd3aaa5baf02563dead1052b51c92eea90616'
    previous = {r['stem']: r for r in read(previous_path)['rows']['baseline']}
    if not smoke:
        small = read(reports / 'trajectory-existing-division-scope-v1-smoke.json')
        assert small['status'] == 'existing_division_scope_audited'
        assert small['source_sha256'] == sha(Path(__file__))
    target.mkdir()
    report = dict(status='freezing_prediction_only_scope', source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT / 'research/trajectory_existing_division_scope_v1.py'),
        dataset_sha256=sha(dataset_path), prior_scoring_sha256=sha(previous_path), records={},
        source_only=True, selection_or_validation_opened=False, ground_truth_opened=False,
        gpu_used=False, model_fitted=False, candidate_changed=False, oracle_graphs_exported=False,
        per_fork_gt_labels_exported=False, authorized_for_submission=False)

    def persist():
        report['seconds'] = time.monotonic() - started
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        plans = {}
        for stem, item in dataset['records'].items():
            scope = read(ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(item['batch'])) / 'MOVIES.json')
            assert any(m['stem'] == stem and m['role'] == 'optimization' for m in scope['movies'])
            baseline_path, initial_path = ROOT / item['baseline_path'], ROOT / item['initial_path']
            assert sha(baseline_path) == item['baseline_sha256']
            harvest = read(reports / ('trajectory-event-source-v1-b' + str(item['batch']) + '-full-harvest.json'))
            assert harvest['status'] == 'verified_backup'
            old = next(r for r in harvest['records'] if r['path'] == stem + '-original/pre-postprocess.json')
            assert sha(initial_path) == old['sha256']
            baseline, initial = read(baseline_path), read(initial_path)
            validate_graph(baseline, 100)
            forks = inventory(initial, baseline)
            plans[stem] = dict(forks=forks, baseline_path=str(baseline_path), baseline_sha256=sha(baseline_path),
                               bytes=baseline_path.stat().st_size)
        chosen = [min((s for s in plans if plans[s]['forks']), key=lambda s: (plans[s]['bytes'], s))] if smoke else sorted(plans)
        plan_path = target / 'PREDICTION_ONLY_SCOPE.json'
        plan_path.write_text(json.dumps({s: plans[s] for s in chosen}, indent=2) + '\n', encoding='utf-8')
        report.update(movies=chosen, scope_sha256=sha(plan_path), all_requested_scopes_frozen_before_gt=True)
        persist()
        helpers = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
        scorer = helpers['load_scorer']()
        divisions = importlib.import_module(scorer.__package__ + '.division_metrics')
        import tracksdata as td
        truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
        manifest = truth_root / 'train_geff_cache_manifest.json'
        assert sha(manifest) == '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files = read(manifest)['files']
        report.update(status='auditing_existing_division_counts', ground_truth_opened=True)
        persist()
        for stem in chosen:
            item = plans[stem]
            selected = [r for r in files if r['relative_path'].startswith(stem + '.geff/')]
            assert len(selected) == 21
            for r in selected:
                assert sha(truth_root / 'train' / r['relative_path']) == r['sha256']
            assert sha(Path(item['baseline_path'])) == item['baseline_sha256']
            graph, mapping = mapped_graph(read(Path(item['baseline_path'])))
            truth = td.graph.IndexedRXGraph.from_geff(str(truth_root / 'train' / (stem + '.geff')))[0]
            scored = divisions.score_divisions(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
            assert len(scored.tp_forks) == previous[stem]['division_tp']
            assert len(scored.fp_forks) == previous[stem]['division_fp']
            assert sum(not v for v in scored.scores.values()) == previous[stem]['division_fn']
            assert not scored.tp_forks.intersection(scored.fp_forks)
            assert scored.tp_forks | scored.fp_forks <= {mapping[p] for p in item['forks']}
            counts = Counter()
            for parent, facts in item['forks'].items():
                category = 'tp' if mapping[parent] in scored.tp_forks else 'fp' if mapping[parent] in scored.fp_forks else 'unscored_not_negative'
                counts[category] += 1
                counts[category + ('_observed_scope' if facts['observed_consecutive_scope'] else '_hard_protected')] += 1
                if facts['observed_consecutive_scope'] and facts['mother_has_single_consecutive_history'] and facts['both_daughters_have_single_consecutive_future']:
                    counts[category + '_with_temporal_context'] += 1
            report['records'][stem] = dict(counts=counts, baseline_counts_reproduced=True)
            persist()
            print(json.dumps(dict(stem=stem, counts=counts)), flush=True)
        report['by_embryo'] = {}
        for embryo in ('44b6', '6bba'):
            count = Counter()
            for stem, record in report['records'].items():
                if stem.startswith(embryo + '_'): count.update(record['counts'])
            report['by_embryo'][embryo] = count
        report['status'] = 'existing_division_scope_audited'
        persist()
    except BaseException as error:
        report.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=1, user_api='blas'):
        main(parser.parse_args().smoke)
