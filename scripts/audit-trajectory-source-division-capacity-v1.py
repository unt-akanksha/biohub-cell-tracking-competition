"""Counts-only source fork coverage in the actual licensed predictor graph."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha


def main():
    started = time.monotonic()
    output = ROOT / 'reports/experiments/trajectory-source-division-capacity-v1.json'
    assert not output.exists()
    scope = ROOT / '.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(scope) == '5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    stems = [m['stem'] for m in json.loads(scope.read_text())['movies']]
    base = ROOT / '.biohub/cache/trajectory-disagreement-source-v1-full-output'
    backup = json.loads((ROOT / 'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert backup['status'] == 'verified_backup'
    for record in backup['records']:
        assert sha(base / record['path']) == record['sha256']
    reference = json.loads((ROOT / 'reports/experiments/trajectory-joint-source-v1-result.json').read_text())
    expected = {r['stem']: r for r in reference['rows']['original']}
    inventory = json.loads((ROOT / 'reports/experiments/trajectory-candidate-supervision-v1.json').read_text())
    truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
    manifest = truth_root / 'train_geff_cache_manifest.json'
    assert sha(manifest) == '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    truth_files = json.loads(manifest.read_text())['files']
    helper = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    scorer = helper['load_scorer']()
    import polars as pl
    import tracksdata as td
    keys = td.DEFAULT_ATTR_KEYS
    records, pooled, embryos = {}, Counter(), defaultdict(Counter)
    for stem in stems:
        selected = [r for r in truth_files if r['relative_path'].startswith(stem + '.geff/')]
        assert len(selected) == 21
        for r in selected:
            assert sha(truth_root / 'train' / r['relative_path']) == r['sha256']
        final = json.loads((base / (stem + '-original') / 'repaired-prediction.json').read_text())
        graph = td.graph.InMemoryGraph()
        for axis in ('z', 'y', 'x'):
            graph.add_node_attr_key(axis, pl.Float64, 0.)
        graph.add_node_attr_key('source_original_id', pl.Int64, -1)
        ids = sorted(final['nodes'], key=int)
        mapped = graph.bulk_add_nodes([dict({k: final['nodes'][i][k] for k in ('t','z','y','x')},
                                           source_original_id=int(i)) for i in ids])
        mapping = dict(zip(map(int, ids), mapped))
        graph.bulk_add_edges([dict(source_id=mapping[e['source_id']], target_id=mapping[e['target_id']])
                              for e in final['edges']])
        truth = td.graph.IndexedRXGraph.from_geff(str(truth_root / 'train' / (stem + '.geff')))[0]
        evaluation = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
        for key in scorer.COUNT_COLUMNS:
            assert getattr(evaluation, key) == expected[stem][key]
        reverse = {int(r[keys.MATCHED_NODE_ID]): int(r['source_original_id'])
                   for r in graph.node_attrs().iter_rows(named=True)
                   if r[keys.MATCHED_NODE_ID] is not None and r[keys.MATCHED_NODE_ID] >= 0}
        assert len(set(reverse.values())) == len(reverse)
        gt_nodes = {int(r[keys.NODE_ID]): r for r in truth.node_attrs().iter_rows(named=True)}
        gt_out = defaultdict(list)
        for edge in truth.edge_attrs().iter_rows(named=True):
            gt_out[int(edge[keys.EDGE_SOURCE])].append(int(edge[keys.EDGE_TARGET]))
        gp = ROOT / '.biohub/cache/trajectory-candidate-supervision-v1' / (stem + '-candidates.npz')
        assert sha(gp) == inventory['candidate_sha256'][stem]
        with np.load(gp, allow_pickle=False) as data:
            groups = dict(data)
        choices = {int(child): set(map(int, groups['parents'][a:b]))
                   for child, a, b in zip(groups['children'], groups['offsets'][:-1], groups['offsets'][1:])}
        with np.load(base / (stem + '-original') / 'raw-candidates.npz', allow_pickle=False) as raw:
            neural_pairs = {(int(a), int(b)) for a, b in raw['edges'][:, :2]}
        final_pairs = {(e['source_id'], e['target_id']) for e in final['edges']}
        counts = Counter()
        for parent, daughters in gt_out.items():
            if len(daughters) != 2:
                continue
            counts['annotated_forks'] += 1
            if any(gt_nodes[d]['t'] != gt_nodes[parent]['t'] + 1 for d in daughters):
                counts['nonconsecutive_forks'] += 1
                continue
            counts['consecutive_forks'] += 1
            if not all(i in reverse for i in (parent, *daughters)):
                counts['missing_at_least_one_matched_cell'] += 1
                continue
            p, a, b = (reverse[i] for i in (parent, *daughters))
            counts['all_three_cells_matched'] += 1
            present = sum((p, d) in final_pairs for d in (a, b))
            counts['final_exact_fork' if present == 2 else 'final_missing_one_or_both_fork_edges'] += 1
            if all((p, d) in neural_pairs for d in (a, b)):
                counts['both_edges_in_neural_proposals'] += 1
            if all(p in choices.get(d, ()) for d in (a, b)):
                counts['both_edges_in_joint_candidate_set'] += 1
                if present != 2:
                    counts['candidate_reachable_missing_forks'] += 1
        counts['official_division_tp'] = evaluation.division_tp
        counts['official_division_fp'] = evaluation.division_fp
        counts['official_division_fn'] = evaluation.division_fn
        records[stem] = dict(counts)
        pooled.update(counts)
        embryos[stem.split('_')[0]].update(counts)
        print(json.dumps(dict(stem=stem, counts=dict(counts))), flush=True)
    result = dict(status='source_division_capacity_audited', per_movie=records, pooled=dict(pooled),
                  by_embryo={k: dict(v) for k, v in embryos.items()}, source_scope_sha256=sha(scope),
                  source_sha256=sha(Path(__file__)), source_only=True, selection_or_validation_opened=False,
                  model_fitted=False, graph_predictions_changed=False, oracle_graphs_exported=False,
                  exact_time_coverage_is_not_official_plus_minus_one_frame_division_recall=True,
                  note='Capacity inventory is not a trained model or evidence for relaxing division thresholds.',
                  elapsed_seconds=time.monotonic() - started)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], pooled=result['pooled'], seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
