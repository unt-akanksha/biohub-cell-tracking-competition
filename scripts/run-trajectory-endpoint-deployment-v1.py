"""Freeze four complete graphs, then score a single fixed endpoint repair."""
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import STEMS, sha, csv_equivalent_graph
from research.public_d4_quality import compare
from research.learned_trajectory_endpoint_v1 import reconnect


def main():
    started = time.monotonic()
    output = ROOT / '.biohub/cache/trajectory-endpoint-deployment-v1-screen'
    target = ROOT / 'reports/experiments/trajectory-endpoint-deployment-v1-result.json'
    if output.exists() or target.exists():
        raise ValueError('Preserve previous staged/completed run')
    base = ROOT / '.biohub/cache/public-d4-full-movie-v1-output'
    prior_path = ROOT / 'reports/experiments/public-d4-full-movie-v1-result.json'
    if sha(prior_path) != '04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b':
        raise ValueError('Frozen parent scores changed')
    v1 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    v2 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v2.py'))
    _, prepared = v1['validate_predictions'](
        base, '61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    inventory = json.loads((ROOT / 'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    for row in inventory['files']:
        if '-original/' in row['path'] and row['path'].endswith(('/pre-postprocess.json', '/raw-candidates.npz')):
            if sha(base / row['path']) != row['sha256']:
                raise ValueError('Frozen parent inference changed')
    names = ('research/learned_trajectory_endpoint_v1.py', 'research/public_d4_quality.py',
             'scripts/run-trajectory-endpoint-deployment-v1.py',
             'reports/experiments/trajectory-endpoint-deployment-v1-design.md')
    pins = {n: sha(ROOT / n) for n in names}
    model_root = ROOT / '.biohub/cache/trajectory-endpoint-deployment-v1'
    model_path = model_root / 'manifest.json'
    model_manifest = json.loads(model_path.read_text())
    model_manifest_sha = sha(model_path)
    if (model_manifest['status'] != 'frozen' or model_manifest['movie_name_used_for_inference']
            or model_manifest['code_sha256'] != sha(ROOT / 'research/learned_trajectory_endpoint_v1.py')
            or model_manifest['design_sha256'] != sha(ROOT / 'reports/experiments/trajectory-endpoint-deployment-v1-design.md')
            or sha(model_root / model_manifest['model_path']) != model_manifest['model_sha256']
            or set(STEMS) & set(model_manifest['optimization_stems'] + model_manifest['calibration_stems'])):
        raise ValueError('Frozen same-code excluded-movie deployment model required')
    with np.load(model_root / model_manifest['model_path'], allow_pickle=False) as data:
        model = {k: data[k] for k in data.files}
    if float(model['threshold']) != model_manifest['threshold']:
        raise ValueError('Frozen deployment threshold changed')
    output.mkdir(parents=True)
    timer = threading.Timer(1200, lambda: os._exit(124)); timer.daemon = True; timer.start()
    receipt = dict(pins=pins, model_manifest_sha256=model_manifest_sha, records={}, ground_truth_used=False)
    for stem in STEMS:
        folder = base / f'{stem}-original'
        pre = json.loads((folder / 'pre-postprocess.json').read_text())
        with np.load(folder / 'raw-candidates.npz', allow_pickle=False) as raw:
            for ident, node in pre['nodes'].items():
                if int(ident) >= len(raw['coords']) or not np.array_equal(
                        raw['coords'][int(ident)], [node[k] for k in ('t', 'z', 'y', 'x')]):
                    raise ValueError('Raw detector identity mismatch')
            graph, details = reconnect(prepared[stem]['original'], raw['coords'], raw['edges'], pre['nodes'], model, float(model['threshold']))
        if csv_equivalent_graph({int(k): v for k, v in graph['nodes'].items()}, graph['edges'], 100) != graph:
            raise ValueError('Invalid complete CSV-equivalent graph')
        path = output / f'{stem}.json'
        path.write_text(json.dumps(graph, sort_keys=True, allow_nan=False))
        receipt['records'][stem] = dict(path=path.name, sha256=sha(path), **details)
        print(json.dumps(dict(event='candidate_persisted', stem=stem,
                             **{k: v for k, v in details.items() if k != 'links'})), flush=True)
    manifest = output / 'prelabel-manifest.json'
    manifest.write_text(json.dumps(receipt, indent=2) + '\n')
    result = dict(run_id='trajectory-endpoint-deployment-v1', receipt=receipt,
                  manifest_sha256=sha(manifest), gpu_hours=0, independently_held_out=False,
                  authorized_for_submission=False,
                  authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    if not sum(r['added_edges'] for r in receipt['records'].values()):
        result.update(status='rejected_no_links', ground_truth_opened=False,
                      diagnostic_gate_passed=False)
    else:
        prior = json.loads(prior_path.read_text())
        truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
        v2['verify_truth_inventory'](truth_root, '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
        scorer = v1['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        rows = []
        for stem in STEMS:
            record = receipt['records'][stem]; path = output / record['path']
            if sha(path) != record['sha256']:
                raise ValueError('Candidate changed after freeze')
            graph = v1['prediction_graph'](json.loads(path.read_text()))
            truth_path = truth_root / 'train' / (stem + '.geff')
            truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
            count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row = dict(scorer.per_sample_metrics(er, count, scorer.node_recall(graph, truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows.append(row)
            print(json.dumps(v1['finite'](dict(event='scored', **row))), flush=True)
        summary = scorer.summarise(rows)
        embryos = {e: scorer.summarise([r for r in rows if r['embryo'] == e]) for e in ('44b6', '6bba')}
        movies = {r['stem']: scorer.summarise([r]) for r in rows}
        comparison = compare(
            {'original': prior['per_movie']['original'], 'corrected': rows},
            {'original': prior['summaries']['original'], 'corrected': summary},
            {'original': prior['by_embryo']['original'], 'corrected': embryos},
            {'original': prior['per_movie_summaries']['original'], 'corrected': movies})
        tp_gain = sum(r['edge_tp'] for r in rows) - sum(r['edge_tp'] for r in prior['per_movie']['original'])
        comparison.update(edge_tp_gain=tp_gain)
        comparison['diagnostic_gate_passed'] &= tp_gain > 0
        result.update(status='diagnostic_pass' if comparison['diagnostic_gate_passed'] else 'rejected_diagnostic',
                      ground_truth_opened=True, per_movie=rows, summary=summary,
                      by_embryo=embryos, per_movie_summaries=movies, comparison=comparison)
    if sha(model_path) != model_manifest_sha:
        raise ValueError('Model manifest changed during evaluation')
    if any(sha(ROOT / n) != h for n, h in pins.items()):
        raise ValueError('Frozen experiment changed during execution')
    result['elapsed_seconds'] = time.monotonic() - started
    target.write_text(json.dumps(v1['finite'](result), indent=2, allow_nan=False) + '\n')
    timer.cancel()
    print(json.dumps(v1['finite']({k: result[k] for k in ('status', 'elapsed_seconds', 'summary', 'comparison') if k in result})), flush=True)


if __name__ == '__main__':
    main()




