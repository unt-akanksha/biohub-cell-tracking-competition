"""CPU rescore persisted bridge arms with current and historical scorers.

Validates all prediction hashes and mutations BEFORE opening any GT. Does not
authorize production: the base association model trained on these movies.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import runpy
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT / 'scripts/verify-focus-bridge-paired.py'))
SOURCES = {
    'current': (ROOT / '.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot', {
        'metrics.py': 'ab11310db0ada78ebb408fcd913bd001aed0dd5f8c1179e0a45dbb7152d6ebde',
        'division_metrics.py': 'ef7472347a06842982bda795f896fd60fd7cb2e429fb87a72c6b9ede50c49d29'}),
    'packaged': (ROOT / '.biohub/cache/focus-replay-scorer', {
        'metrics.py': '31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83',
        'division_metrics.py': 'd1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be'}),
}


def load_scorer(name):
    folder, hashes = SOURCES[name]
    for filename, digest in hashes.items():
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} scorer source hash changed: {filename}')
    package_name = f'_focus_replay_{name}'
    package = ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    return importlib.import_module(package_name + '.metrics')


def validate_arms(root):
    manifest = json.loads((root / 'prelabel_graph_manifest.json').read_text())
    if set(manifest) != VERIFY['STEMS']:
        raise ValueError('complete four-movie paired manifest required')
    prepared = {}
    for stem in sorted(manifest):
        arms = {}
        for arm in ('control', 'bridge'):
            path = root / 'paired_graphs' / f'{stem}-{arm}.json'
            if VERIFY['sha'](path) != manifest[stem]['graphs'][arm]['sha256']:
                raise ValueError('prediction graph hash mismatch')
            arms[arm] = json.loads(path.read_text())
        VERIFY['check_mutation'](arms['control'], arms['bridge'])
        prepared[stem] = arms
    return prepared


def prediction_graph(payload):
    import polars as pl
    import tracksdata as td
    graph = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.0)
    ids = sorted(payload['nodes'], key=int)
    mapped = graph.bulk_add_nodes([{k: payload['nodes'][i][k] for k in ('t', 'z', 'y', 'x')} for i in ids])
    mapping = dict(zip(map(int, ids), mapped))
    if payload['edges']:
        graph.bulk_add_edges([dict(source_id=mapping[e['source_id']], target_id=mapping[e['target_id']])
                              for e in payload['edges']])
    return graph


def replay(root, truth_root):
    # No GT is touched before all eight graphs pass transport/mutation checks.
    prepared = validate_arms(root)
    scorers = {name: load_scorer(name) for name in SOURCES}
    import tracksdata as td
    from geff import GeffMetadata
    result = {'authoritative_scorer_commit': '075fc5f5a52d11077f9dc2b074644618f26939e2',
              'authorized_for_submission': False, 'authorized_for_production_promotion': False,
              'validation_scope': 'base-training diagnostic', 'scorers': {}}
    for name, scorer in scorers.items():
        rows = {arm: [] for arm in ('control', 'bridge')}
        for stem, arms in prepared.items():
            for arm, payload in arms.items():
                path = truth_root / f'{stem}.geff'
                truth = td.graph.IndexedRXGraph.from_geff(str(path))[0]
                graph = prediction_graph(payload)
                counts = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.0)
                total = float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
                row = scorer.per_sample_metrics(counts, total, scorer.node_recall(graph, truth))
                rows[arm].append(dict(row, stem=stem, embryo=stem.split('_')[0]))
                print(json.dumps(dict(scorer=name, arm=arm, **rows[arm][-1])), flush=True)
        summary = {arm: scorer.summarise(values) for arm, values in rows.items()}
        result['scorers'][name] = dict(per_movie=rows, summaries=summary,
            by_embryo={arm: {embryo: scorer.summarise([r for r in values if r['embryo'] == embryo])
                             for embryo in ('44b6', '6bba')} for arm, values in rows.items()})
    current = result['scorers']['current']
    c, b = current['summaries']['control'], current['summaries']['bridge']
    deltas = {x['stem']: y['adj_edge_jaccard'] - x['adj_edge_jaccard']
              for x, y in zip(current['per_movie']['control'], current['per_movie']['bridge'])}
    tp_gain = sum(r['edge_tp'] for r in current['per_movie']['bridge']) - sum(r['edge_tp'] for r in current['per_movie']['control'])
    div_ok = b['division_jaccard'] >= c['division_jaccard'] or all(math.isnan(s['division_jaccard']) for s in (c, b))
    result.update(current_official_score_delta=b['score']-c['score'],
        per_movie_adjusted_delta=deltas,
        diagnostic_gate_passed=bool(b['score'] > c['score'] and min(deltas.values()) >= 0 and tp_gain > 0 and div_ok))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--truth-root', type=Path, default=ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train')
    args = parser.parse_args()
    print(json.dumps(replay(args.output_root, args.truth_root), indent=2))
