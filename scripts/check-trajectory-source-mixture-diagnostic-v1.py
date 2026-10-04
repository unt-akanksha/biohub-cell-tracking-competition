"""Freeze generic two-expert outputs; reuse scores only for identical graphs."""
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha, STEMS, csv_equivalent_graph
from research.learned_trajectory_endpoint_v1 import reconnect as endpoint
from research.trajectory_source_mixture_v1 import reconnect as mixture


def main():
    start = time.monotonic()
    output = ROOT / '.biohub/cache/trajectory-source-mixture-v1-diagnostic'
    report = ROOT / 'reports/experiments/trajectory-source-mixture-v1-diagnostic.json'
    if output.exists() or report.exists():
        raise ValueError('Preserve completed mixture check')
    prior_path = ROOT / 'reports/experiments/learned-trajectory-endpoint-v1-result.json'
    if sha(prior_path) != 'f48299001d0733c601b8356c2977dffd2ac79c9cd829252d7b504d0741e0a8d1':
        raise ValueError('Cross-source diagnostic receipt changed')
    prior = json.loads(prior_path.read_text())
    models_root = ROOT / '.biohub/cache/learned-trajectory-endpoint-v1-models'
    if sha(models_root / 'manifest.json') != '57236f97756e7287204c1db32b084e0bbbe48315012b3117af444ddfa7296aa3':
        raise ValueError('Frozen experts changed')
    manifest = json.loads((models_root / 'manifest.json').read_text()); models = []
    for target in ('6bba', '44b6'):
        row = manifest['models'][target]; path = models_root / row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Model changed')
        with np.load(path, allow_pickle=False) as data:
            models.append({k:data[k] for k in data.files})
    base = ROOT / '.biohub/cache/public-d4-full-movie-v1-output'
    helper = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    _, payloads = helper['validate_predictions'](base, '61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683')
    inventory = json.loads((ROOT / 'reports/experiments/public-d4-full-movie-v1-artifact-manifest.json').read_text())
    for r in inventory['files']:
        if '-original/' in r['path'] and r['path'].endswith(('/raw-candidates.npz', '/pre-postprocess.json')):
            if sha(base / r['path']) != r['sha256']:
                raise ValueError('Original detector evidence changed')
    output.mkdir(parents=True); receipts = {}; identical = True
    for stem in STEMS:
        folder = base / (stem + '-original')
        pre = json.loads((folder / 'pre-postprocess.json').read_text())
        with np.load(folder / 'raw-candidates.npz', allow_pickle=False) as raw:
            graph, details = mixture(payloads[stem]['original'], raw['coords'], raw['edges'], pre['nodes'], models, endpoint)
        if csv_equivalent_graph({int(k):v for k,v in graph['nodes'].items()}, graph['edges'], 100) != graph:
            raise ValueError('Invalid mixture graph')
        path = output / (stem + '.json'); path.write_text(json.dumps(graph, sort_keys=True, allow_nan=False))
        old_record = prior['receipt']['records'][stem]
        old_path = ROOT / '.biohub/cache/learned-trajectory-endpoint-v1' / old_record['path']
        if sha(old_path) != old_record['sha256']:
            raise ValueError('Previously scored graph changed')
        same = graph == json.loads(old_path.read_text()); identical &= same
        receipts[stem] = dict(path=path.name, sha256=sha(path), identical_to_scored_graph=same, **details)
        print(json.dumps(dict(stem=stem, identical_to_scored_graph=same,
                             added_edges=details['added_edges'], expert_added_edges=details['expert_added_edges'])), flush=True)
    result = dict(status='identical_diagnostic_pass' if identical else 'requires_new_full_diagnostic',
                  receipts=receipts, source_sha256=sha(Path(__file__)),
                  mixture_sha256=sha(ROOT / 'research/trajectory_source_mixture_v1.py'),
                  motion_sha256=sha(ROOT / 'research/learned_trajectory_endpoint_v1.py'),
                  design_sha256=sha(ROOT / 'reports/experiments/trajectory-source-mixture-v1-design.md'),
                  reused_score_receipt_sha256=sha(prior_path) if identical else None,
                  verified_identical_graph_summary=prior['summary'] if identical else None,
                  ground_truth_opened=False, authorized_for_submission=False,
                  independently_held_out=False, elapsed_seconds=time.monotonic()-start)
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
