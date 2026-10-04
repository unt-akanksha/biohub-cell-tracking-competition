"""Measure input-domain mismatch; no fitted model, held-out labels, or scoring."""
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from research.learned_division_recovery import discover_division_recovery_candidates
import runpy
# Load the pure NumPy file directly; the historical package initializer imports Torch.
_context = runpy.run_path(str(ROOT / 'research/temporal_contrastive/graph_context_features.py'))
physical_nodes, context_tokens = _context['physical_nodes'], _context['context_tokens']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(rows=0)
    return dict(rows=len(a), mean=float(a.mean()), minimum=float(a.min()),
                maximum=float(a.max()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p99=float(np.quantile(a, .99)))


def main():
    started = time.monotonic()
    target = ROOT / 'reports/experiments/division-context-domain-v1-result.json'
    if target.exists():
        raise ValueError('Preserve completed diagnostic')
    split_path = ROOT / 'research/graph_context_fresh_split_v3.json'
    if sha(split_path) != '8d53a55217be88efc895ae9bf0bf378a3a4acb6c42437836a342d888cc9296da':
        raise ValueError('Split changed')
    roles = {r['stem']: r['role'] for r in json.loads(split_path.read_text())['stem_roles']}
    data = ROOT / '.biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1'
    source = data / 'graph_context_relational_patch_manifest.json'
    if sha(source) != '2e5c4c46b11ff1480194c29e88c0e16b0ebf2f0fd65b2fc29548d26fee599bd9':
        raise ValueError('Frozen source dataset changed')
    rows = [r for r in json.loads(source.read_text())['records'] if roles[r['stem']] == 'optimization']
    training = defaultdict(list)
    for row in rows:
        path = data / row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Optimization shard changed')
        with np.load(path, allow_pickle=False) as a:
            mask = a['graph_context_mask']
            if mask.shape != (row['rows'], 43) or not mask[:, :3].all():
                raise ValueError('Invalid source context')
            training[row['embryo']].extend(mask.sum(axis=1).tolist())
    parents = ROOT / '.biohub/cache/public-d4-full-movie-v1-output'
    inventory = json.loads((ROOT / 'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    pins = {r['path']: r['sha256'] for r in inventory['files']}
    deployment = {}
    for stem in ('44b6_81c256f0', '6bba_f1fde7e0'):
        if roles[stem] != 'optimization':
            raise ValueError('Diagnostic may only inspect optimization movies')
        relative = f'{stem}-original/prediction.json'
        if sha(parents / relative) != pins[relative]:
            raise ValueError('Frozen public prediction changed')
        graph = json.loads((parents / relative).read_text())
        nodes = {int(k): v for k, v in graph['nodes'].items()}
        candidates = discover_division_recovery_candidates(nodes, graph['edges'])
        physical = physical_nodes(nodes)
        points = defaultdict(list)
        for ident, (t, xyz) in physical.items():
            points[t].append(xyz)
        trees = {t: cKDTree(x) for t, x in points.items()}
        counts, eligible_counts = [], []
        for index, c in enumerate(candidates):
            t, position = physical[c.parent_id]
            anchors = {c.parent_id, c.existing_child_id, c.second_child_id}
            count = 3
            for dt in (-2, -1, 0, 1, 2):
                total = len(trees[t + dt].query_ball_point(position, 30.)) if t + dt in trees else 0
                total -= sum(physical[i][0] == t + dt and float(np.linalg.norm(physical[i][1] - position)) <= 30. for i in anchors)
                count += min(8, total)
            if index < 5:
                _, exact = context_tokens(physical, parent_id=c.parent_id,
                    existing_child_id=c.existing_child_id, proposed_child_id=c.second_child_id)
                if int(exact.sum()) != count:
                    raise ValueError('Accelerated count disagrees with actual inference')
            counts.append(count)
            if c.biological_geometry_score >= 3:
                eligible_counts.append(count)
        ref = training[stem.split('_')[0]]
        deployment[stem] = dict(all_candidates=summary(counts), geometry_eligible=summary(eligible_counts),
            fraction_above_optimization_p99=float(np.mean(np.asarray(counts) > np.quantile(ref, .99))) if counts else None,
            prediction_sha256=pins[relative], candidate_labels_opened=False)
        print(json.dumps(dict(stem=stem, **deployment[stem])), flush=True)
    result = dict(status='complete_input_domain_diagnostic', optimization= {e: summary(v) for e, v in training.items()},
        deployment=deployment, elapsed_seconds=time.monotonic()-started, source_manifest_sha256=sha(source),
        source_sha256=sha(__file__), optimization_shards_read=len(rows),
        selection_or_audit_shards_opened=False, division_targets_opened=False,
        models_fitted=False, gpu_hours=0, authorized_for_submission=False,
        limitation='Input distribution evidence, not proof of model reliance, causality, or improvement')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
