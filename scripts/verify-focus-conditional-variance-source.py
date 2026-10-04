"""Replay source graph transport, topology, controls and metric arithmetic.

The source run performs fresh official GT matching. This verifier does not claim
an additional independent matching pass; it replays predictions and aggregates.
"""
import json
import os
from pathlib import Path
import runpy
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_conditional_variance_linking import link


def verify():
    runner = runpy.run_path(str(ROOT / 'scripts/score-focus-conditional-variance.py'))
    sha, run, base, source = (runner[k] for k in ('sha', 'RUN', 'BASE', 'SOURCE'))
    path = ROOT / f'reports/experiments/{run}-result.json'
    result = json.loads(path.read_text())
    if any(sha(ROOT / p) != digest for p, digest in result['source_hashes'].items()):
        raise ValueError('Actual executed source method changed')
    training_receipt, model = runpy.run_path(str(ROOT / 'scripts/verify-focus-conditional-variance.py'))['verify']()
    receipt_path = ROOT / 'reports/experiments/focus-conditional-variance-v1-verification.json'
    if (json.loads(receipt_path.read_text()) != training_receipt
            or sha(receipt_path) != result['training_verification_sha256']
            or model != result['model'] or training_receipt['final_model_sha256'] != result['model_sha256']):
        raise ValueError('Exact actual training-only model required')
    reference_path = ROOT / 'reports/experiments/focus-source-flow-v1-result.json'
    constant_path = ROOT / 'reports/experiments/focus-conditional-motion-source-v1-result.json'
    if (sha(reference_path) != base['REFERENCE_SHA']
            or sha(constant_path) != '998e43b168cd4402a081349d48b0cea9a0843783e72f22000c5e7c7c4ce8b400'):
        raise ValueError('Frozen control results changed')
    reference = json.loads(reference_path.read_text())
    constant = json.loads(constant_path.read_text())
    stems = reference['contract']['source_stems']
    cache = ROOT / '.biohub/cache' / run
    manifest_path = cache / 'prelabel_manifest.json'
    if sha(manifest_path) != result['prelabel_manifest_sha256']:
        raise ValueError('Actual prelabel manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['source_hashes'] != result['source_hashes']
            or manifest['model_sha256'] != result['model_sha256']
            or [r['stem'] for r in manifest['records']] != stems
            or len(stems) != 8 or manifest['all_predictions_saved_before_source_gt'] is not True):
        raise ValueError('Complete frozen eight-movie prediction scope required')
    total_nodes, total_edges = 0, 0
    for record in manifest['records']:
        stem = record['stem']
        graph_path = cache / (stem + '.npz')
        sample_path = ROOT / '.biohub/cache/kernel-outputs/focus-source-flow-v1/focus_source_flow/outputs' / stem / 'sampled_flow.npz'
        if sha(graph_path) != record['candidate_sha256'] or sha(sample_path) != record['sample_sha256']:
            raise ValueError('Actual graph or source motion hash changed')
        with np.load(graph_path, allow_pickle=False) as saved:
            coords, edges = saved['coords'].copy(), saved['edges'].copy()
        with np.load(sample_path, allow_pickle=False) as saved:
            if not np.array_equal(coords, saved['coords']):
                raise ValueError('Candidate changed raw node geometry')
            predicted = link(coords, saved['backward_um'], model)
        expected = np.asarray([(s, d) for s, d, _ in predicted], dtype=np.int64).reshape(-1, 2)
        if (not np.array_equal(edges, expected) or edges.dtype != np.int64
                or len(coords) != record['nodes'] or len(edges) != record['edges']
                or sorted(set(coords[:, 0])) != list(range(100))):
            raise ValueError('Complete actual graph replay or movie coverage differs')
        total_nodes += len(coords)
        total_edges += len(edges)
    metric = source['SCORER']['load_scorer'](ROOT / '.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    rows = result['per_movie']
    if set(rows) != {'parent', 'control', 'constant', 'candidate'} or any([r['stem'] for r in arm] != stems for arm in rows.values()):
        raise ValueError('Complete paired control/candidate coverage required')
    summaries = {arm: metric.summarise(values) for arm, values in rows.items()}
    if summaries != result['summaries'] or result['by_embryo'] != {a: {'6bba': s} for a, s in summaries.items()}:
        raise ValueError('Pooled and per-embryo arithmetic differs')
    comparison = base['comparison'](rows, summaries, reference)
    if (comparison != result['comparison'] or rows['constant'] != constant['per_movie']['candidate']
            or summaries['constant'] != constant['summaries']['candidate']):
        raise ValueError('Exact previous source controls and gates required')
    delta = {k: summaries['candidate'][k] - summaries['constant'][k] for k in ('score', 'edge_jaccard', 'division_tp')}
    conditions = dict(original_source_and_flow_gates=comparison['source_gate_passed'],
                      improves_previous_calibrated_score=delta['score'] > 0,
                      improves_previous_calibrated_raw_jaccard=delta['edge_jaccard'] > 0,
                      previous_true_divisions_preserved=delta['division_tp'] >= 0)
    if (conditions != result['conditions'] or delta != result['previous_calibrated_deltas']
            or result['source_gate_passed'] != all(conditions.values())
            or result['gpu_seconds'] != 0 or result['new_target_movies_opened'] != 0
            or result['authorized_for_submission'] is not False):
        raise ValueError('Source decision or resource scope differs')
    return dict(status='verified_conditional_variance_source_artifacts', result_sha256=sha(path),
                nodes_replayed=total_nodes, edges_replayed=total_edges, complete_movies=8,
                candidate_summary=summaries['candidate'], previous_calibrated_deltas=delta,
                conditions=conditions, source_gate_passed=all(conditions.values()),
                authorized_for_submission=False,
                verification_scope='Exact predictions/transport, unchanged controls and official aggregate arithmetic; no additional GT matching pass')


if __name__ == '__main__':
    target = ROOT / 'reports/experiments/focus-conditional-variance-source-v1-verification.json'
    if target.exists():
        raise ValueError('Never overwrite completed verification')
    receipt = verify()
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(receipt, indent=2))
