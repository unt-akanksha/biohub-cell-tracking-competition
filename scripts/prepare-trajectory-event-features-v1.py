"""Freeze current-candidate source graphs and edge features before source labels."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_structured_release_v1 import load_portable
from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_candidate_ranker_v1 import FEATURES, features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    started = time.monotonic()
    name = 'trajectory-event-source-v1-b' + str(args.batch)
    base = ROOT / '.biohub/cache' / (name + '-full-output')
    target = ROOT / '.biohub/cache' / (name + '-features')
    assert not target.exists()
    plan = json.loads((ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').read_text())
    record = next(b for b in plan['batches'] if b['index'] == args.batch)
    scope = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(args.batch)) / 'MOVIES.json'
    assert sha(scope) == record['plan_sha256']
    build = json.loads((ROOT / 'reports/experiments' / (name + '-build.json')).read_text())
    terminal = json.loads((base / 'result.json').read_text())
    assert terminal['status'] == 'complete_prelabel_predictions' and terminal['mode'] == 'full'
    assert terminal['contract_sha256'] == build['contract_sha256'] and terminal['inputs_unchanged']
    assert not terminal['ground_truth_opened'] and set(terminal['movies']) == set(record['movies'])
    backup = json.loads((ROOT / 'reports/experiments' / (name + '-full-harvest.json')).read_text())
    assert backup['status'] == 'verified_backup'
    assert {f.relative_to(base).as_posix() for f in base.rglob('*') if f.is_file()} == {r['path'] for r in backup['records']}
    for row in backup['records']:
        assert sha(base / row['path']) == row['sha256']
    portable_root = ROOT / '.biohub/cache/trajectory-structured-portable-v1'
    module, weights = load_portable(portable_root)
    model = json.loads((portable_root / 'structured-trajectory-model.json').read_text())
    for filename in ('trajectory_candidate_inventory_v1.py', 'trajectory_candidate_ranker_v1.py'):
        assert sha(ROOT / 'research' / filename) == model['inference_source_sha256'][filename]
    target.mkdir()
    records = {}
    for stem in record['movies']:
        row = terminal['movies'][stem]['original']
        assert row['frames'] == 100
        folder = base / (stem + '-original')
        ar2_path = folder / 'repaired-prediction.json'
        assert sha(ar2_path) == row['repaired_sha256']
        initial = json.loads((folder / 'pre-postprocess.json').read_text())
        ar2 = json.loads(ar2_path.read_text())
        validate_graph(ar2, 100)
        with np.load(folder / 'raw-candidates.npz', allow_pickle=False) as raw:
            final, details = module.refine(initial, ar2, raw['coords'], raw['edges'], weights)
            validate_graph(final, 100)
            groups = candidates(initial, final)
            matrix = features(initial, final, groups, raw['edges'])
        gp, fp, pp = (target / (stem + suffix) for suffix in ('-groups.npz', '-features.npz', '-prediction.json'))
        np.savez_compressed(gp, **groups)
        np.savez_compressed(fp, features=matrix, feature_names=np.asarray(FEATURES))
        pp.write_text(json.dumps(final, sort_keys=True, allow_nan=False), encoding='utf-8')
        records[stem] = dict(groups_sha256=sha(gp), features_sha256=sha(fp), prediction_sha256=sha(pp),
                             collection_graph_sha256=sha(ar2_path), queries=len(groups['children']),
                             candidate_edges=len(groups['parents']), structured_assignment=details)
        print(json.dumps(dict(stem=stem, queries=len(groups['children']), edges=len(groups['parents']),
                             structured_swaps=details['changed_edges'])), flush=True)
    result = dict(status='current_candidate_source_features_frozen', per_movie=records, source_scope_sha256=sha(scope),
                  terminal_sha256=sha(base / 'result.json'), model_sha256=sha(portable_root / 'structured-trajectory-model.json'),
                  source_sha256=sha(Path(__file__)), ground_truth_opened=False,
                  selection_or_validation_opened=False, model_applied=True, model_fitted=False,
                  no_image_features_extracted=True, authorized_for_submission=False,
                  seconds=time.monotonic() - started)
    (target / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'reports/experiments' / (name + '-features.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'])))


if __name__ == '__main__':
    main()
