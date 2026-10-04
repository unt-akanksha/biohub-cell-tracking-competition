"""Find supervised candidate coverage before choosing an association experiment.

Original public baseline only; four previously exposed complete movies.
No candidate generation, fit, threshold tuning, score promotion or GPU access.
"""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha, STEMS
from research.public_edge_supervision import classify_edge, mapped_pairs


def graph_and_map(payload):
    graph = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.)
    graph.add_node_attr_key('original_candidate_id', pl.Int64, -1)
    keys = sorted(payload['nodes'], key=int)
    graph_ids = graph.bulk_add_nodes([
        dict(original_candidate_id=int(k),
             **{a: payload['nodes'][k][a] for a in ('t', 'z', 'y', 'x')})
        for k in keys])
    mapping = dict(zip(map(int, keys), graph_ids))
    graph.bulk_add_edges([dict(source_id=mapping[e['source_id']],
                               target_id=mapping[e['target_id']])
                          for e in payload['edges']])
    attrs = graph.node_attrs()
    observed = dict(zip(attrs['original_candidate_id'].to_list(),
                        attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()))
    if mapping != observed:
        raise ValueError('Original-to-runtime node ID mapping failed')
    return graph


def main():
    start = time.perf_counter()
    output = ROOT / 'reports/experiments/public-ordinary-edge-coverage-v1-result.json'
    if output.exists():
        raise ValueError('Preserve completed audit')
    v1 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    v2 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v2.py'))
    v1['load_scorer']()  # Require pinned authoritative implementation.
    cache = ROOT / '.biohub/cache/public-d4-full-movie-v1-output'
    _, payloads = v1['validate_predictions'](
        cache, '61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
    v2['verify_truth_inventory'](
        truth_root, '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
    manifest = json.loads((ROOT / 'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    original_score_path = ROOT / 'reports/experiments/public-d4-full-movie-v1-result.json'
    if sha(original_score_path) != '04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b':
        raise ValueError('Original score receipt changed')
    old_rows = {r['stem']: r for r in json.loads(original_score_path.read_text())['per_movie']['original']}
    for row in manifest['files']:
        if '-original/' in row['path'] and row['path'].endswith(('raw-candidates.npz', 'pre-postprocess.json')):
            if sha(cache / row['path']) != row['sha256']:
                raise ValueError('Frozen raw candidate input changed')
    results = {}
    for stem in STEMS:
        payload = payloads[stem]['original']
        path = cache / f'{stem}-original'
        pre = json.loads((path / 'pre-postprocess.json').read_text())
        genuine = set(map(int, pre['nodes'])) & set(map(int, payload['nodes']))
        raw = np.load(path / 'raw-candidates.npz', allow_pickle=False)
        truth = td.graph.IndexedRXGraph.from_geff(str(truth_root / 'train' / (stem + '.geff')))[0]
        graph = graph_and_map(payload)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            graph.match(truth, matching=DistanceMatching(max_distance=7., scale=(1.625, .40625, .40625)))
        attrs = graph.node_attrs()
        matches = dict(zip(attrs['original_candidate_id'].to_list(),
                           attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()))
        te = truth.edge_attrs()
        truth_edges = set(zip(te[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                              te[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()))
        truth_out = {a for a, _ in truth_edges}
        truth_in = {b for _, b in truth_edges}
        final_edges = {(e['source_id'], e['target_id']) for e in payload['edges']}
        final_pairs = mapped_pairs(final_edges, matches)
        true_final = final_pairs & truth_edges
        if len(true_final) != old_rows[stem]['edge_tp']:
            raise ValueError('Mapping does not replay official true positive count')
        candidate_edges = {(int(a), int(b)): float(p) for a, b, p, _ in raw['edges']
                           if int(a) in genuine and int(b) in genuine}
        labels = {e: classify_edge(*e, matches, truth_edges, truth_out, truth_in)
                  for e in candidate_edges}
        raw_true_pairs = mapped_pairs([e for e, y in labels.items() if y == 1], matches)
        matched_truth = {v for v in matches.values() if v not in (None, -1)}
        missing = truth_edges - true_final
        endpoints_present = {e for e in missing if set(e) <= matched_truth}
        ordinary_truth = {e for e in truth_edges
                          if sum(a == e[0] for a, _ in truth_edges) == 1}
        result = dict(raw_edges=len(raw['edges']), genuine_retained_candidate_edges=len(candidate_edges),
                      supervision={str(y): sum(v == y for v in labels.values()) for y in (-1, 0, 1)},
                      official_baseline_edge_tp=len(true_final), official_baseline_edge_fn=len(missing),
                      missing_truth_edges_both_final_endpoints_matched=len(endpoints_present),
                      missing_truth_edges_supported_by_genuine_raw_edge=len(raw_true_pairs - true_final),
                      missing_ordinary_truth_edges_supported_by_genuine_raw_edge=len((raw_true_pairs - true_final) & ordinary_truth),
                      missing_truth_edges_with_unmatched_endpoint=len(missing - endpoints_present),
                      raw_positive_alternatives_not_in_final=sum(y == 1 and e not in final_edges for e, y in labels.items()))
        results[stem] = result
        print(json.dumps(dict(stem=stem, **result)), flush=True)
        raw.close()
    result = dict(run_id='public-ordinary-edge-coverage-v1', status='complete_diagnostic',
                  elapsed_seconds=time.perf_counter() - start, source_sha256=sha(Path(__file__)),
                  helper_sha256=sha(ROOT / 'research/public_edge_supervision.py'),
                  per_movie=results, gpu_hours=0, candidate_generated=False,
                  independently_held_out=False, authorized_for_submission=False,
                  warning='Previously exposed training movies; public model training overlaps. Coverage is not validation gain.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
