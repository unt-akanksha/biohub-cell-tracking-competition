"""Frozen, CPU-only complete-movie screen. Never submits or tunes from scores."""
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import STEMS, sha, csv_equivalent_graph
from research.public_d4_quality import compare
from research.public_supported_bridge_v2 import bridge


def main():
    import numpy as np
    started = time.monotonic()
    output = ROOT / '.biohub/cache/public-supported-bridge-v2'
    target = ROOT / 'reports/experiments/public-supported-bridge-v2-result.json'
    if output.exists() or target.exists():
        raise ValueError('Preserve previous staged/completed run')
    base = ROOT / '.biohub/cache/public-d4-full-movie-v1-output'
    prior_path = ROOT / 'reports/experiments/public-d4-full-movie-v1-result.json'
    if sha(prior_path) != '04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b':
        raise ValueError('Frozen parent scores changed')
    v1 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    v2 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v2.py'))
    _, prepared = v1['validate_predictions'](base, '61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    inventory = json.loads((ROOT / 'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    for row in inventory['files']:
        if row['path'].endswith(('/pre-postprocess.json', '/raw-candidates.npz')):
            if sha(base / row['path']) != row['sha256']:
                raise ValueError('Frozen parent inference changed')
    pins = {n: sha(ROOT / n) for n in (
        'research/public_supported_bridge_v2.py', 'research/public_d4_quality.py',
        'scripts/run-public-supported-bridge-v2.py',
        'reports/experiments/public-supported-bridge-v2-design.md')}
    output.mkdir(parents=True)
    timer = threading.Timer(1200, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    receipt = dict(pins=pins, records={}, ground_truth_used=False)
    for stem in STEMS:
        receipt['records'][stem] = {}
        for parent in ('original', 'corrected'):
            folder = base / f'{stem}-{parent}'
            pre = json.loads((folder / 'pre-postprocess.json').read_text())
            with np.load(folder / 'raw-candidates.npz') as raw:
                for ident, node in pre['nodes'].items():
                    if int(ident) >= len(raw['coords']) or not np.array_equal(
                        raw['coords'][int(ident)], [node[k] for k in ('t', 'z', 'y', 'x')]):
                        raise ValueError('Raw detector index does not match original graph ID')
                graph, details = bridge(prepared[stem][parent], raw['coords'], raw['edges'], pre['nodes'])
            if csv_equivalent_graph({int(k): v for k, v in graph['nodes'].items()}, graph['edges'], 100) != graph:
                raise ValueError('Candidate is not a complete valid movie graph')
            path = output / f'{stem}-{parent}.json'
            path.write_text(json.dumps(graph, sort_keys=True, allow_nan=False))
            row = dict(path=path.name, sha256=sha(path), **details)
            receipt['records'][stem][parent] = row
            print(json.dumps(dict(event='candidate_persisted', stem=stem, parent=parent,
                                 **{k: v for k, v in details.items() if k != 'paths'})), flush=True)
    manifest = output / 'prelabel-manifest.json'
    manifest.write_text(json.dumps(receipt, indent=2) + '\n')
    # Only after all eight predictions are immutable do we open any truth.
    additions = sum(r['added_nodes'] for group in receipt['records'].values() for r in group.values())
    result = dict(run_id='public-supported-bridge-v2', receipt=receipt,
        manifest_sha256=sha(manifest), gpu_hours=0, independently_held_out=False,
        authorized_for_submission=False, authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    if not additions:
        result.update(status='rejected_no_paths', ground_truth_opened=False,
                      diagnostic_gate_passed=False)
    else:
        prior = json.loads(prior_path.read_text())
        truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
        v2['verify_truth_inventory'](truth_root, '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
        scorer = v1['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        rows = {p: [] for p in ('original', 'corrected')}
        for stem in STEMS:
            for parent in rows:
                record = receipt['records'][stem][parent]
                path = output / record['path']
                if sha(path) != record['sha256']:
                    raise ValueError('Candidate changed after freeze')
                graph = v1['prediction_graph'](json.loads(path.read_text()))
                truth_path = truth_root / 'train' / (stem + '.geff')
                truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
                er = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
                count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
                row = dict(scorer.per_sample_metrics(er, count, scorer.node_recall(graph, truth)),
                           stem=stem, embryo=stem.split('_')[0])
                rows[parent].append(row)
                print(json.dumps(v1['finite'](dict(event='scored', parent=parent, **row))), flush=True)
        summaries = {p: scorer.summarise(v) for p, v in rows.items()}
        embryos = {p: {e: scorer.summarise([r for r in v if r['embryo'] == e])
                       for e in ('44b6', '6bba')} for p, v in rows.items()}
        movies = {p: {r['stem']: scorer.summarise([r]) for r in v} for p, v in rows.items()}
        comparisons = {}
        for parent in rows:
            comparison = compare(
                {'original': prior['per_movie']['original'], 'corrected': rows[parent]},
                {'original': prior['summaries']['original'], 'corrected': summaries[parent]},
                {'original': prior['by_embryo']['original'], 'corrected': embryos[parent]},
                {'original': prior['per_movie_summaries']['original'], 'corrected': movies[parent]})
            tp_gain = sum(r['edge_tp'] for r in rows[parent]) - sum(r['edge_tp'] for r in prior['per_movie'][parent])
            own_deltas = {s: movies[parent][s]['score'] - prior['per_movie_summaries'][parent][s]['score'] for s in STEMS}
            own_pass = min(own_deltas.values()) >= 0 and summaries[parent]['score'] > prior['summaries'][parent]['score']
            comparison.update(added_edge_tp_against_parent=tp_gain,
                              own_parent_per_movie_delta=own_deltas, own_parent_passed=own_pass)
            comparison['diagnostic_gate_passed'] &= tp_gain > 0 and own_pass
            comparisons[parent] = comparison
        result.update(status='complete', ground_truth_opened=True, per_movie=rows,
                      summaries=summaries, by_embryo=embryos, per_movie_summaries=movies,
                      comparisons=comparisons)
    if any(sha(ROOT / name) != digest for name, digest in pins.items()):
        raise ValueError('Frozen scientific code changed')
    result['elapsed_seconds'] = time.monotonic() - started
    target.write_text(json.dumps(v1['finite'](result), indent=2, allow_nan=False) + '\n')
    timer.cancel()
    print(json.dumps(v1['finite']({k: v for k, v in result.items()
        if k in ('status', 'summaries', 'comparisons', 'elapsed_seconds')})), flush=True)


if __name__ == '__main__':
    main()
