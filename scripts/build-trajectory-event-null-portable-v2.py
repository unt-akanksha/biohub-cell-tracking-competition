"""Package exact null-dominance inference without project imports or live edits."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_event_release_v1 import check_event_stage


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def portable_source():
    original = ROOT / '.biohub/cache/trajectory-event-fork16-vectorized-portable-v1/event-trajectory.py'
    assert sha(original) == 'eac9ee4f7ed8b5192167e94a39910e8343ff1e1f90880f95dd54360ef9abc161'
    helper = ROOT / 'research/trajectory_event_null_dominance_v1.py'
    tree = ast.parse(original.read_text(encoding='utf-8'))
    functions = [n for n in ast.parse(helper.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)]
    assert len(functions) == 1 and functions[0].name == 'allowed_options'
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'allowed_options']
    assert len(matches) == 1
    matches[0].name = 'fork_pruning'
    tree.body.insert(tree.body.index(matches[0]) + 1, functions[0])
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def main(smoke):
    started = time.monotonic()
    name = 'trajectory-event-null-portable-v2-' + ('smoke' if smoke else 'full')
    target = ROOT / '.biohub/cache' / name
    receipt = ROOT / 'reports/experiments' / (name + '.json')
    assert not target.exists() and not receipt.exists(), 'Inspect existing run; do not overwrite'
    proof_path = ROOT / 'reports/experiments/trajectory-event-public-cache-v2-full.json'
    proof = read(proof_path)
    assert proof['status'] == 'paired_public_runtime_verified' and len(proof['records']) == 4
    replay_path = ROOT / 'reports/experiments/trajectory-event-null-portable-v1-full.json'
    assert sha(replay_path) == proof['source_replay_sha256']
    replay = read(replay_path)
    helper_sha = sha(ROOT / 'research/trajectory_event_null_dominance_v1.py')
    assert helper_sha == replay['helper_sha256']
    code = portable_source()
    if not smoke:
        small = read(ROOT / 'reports/experiments/trajectory-event-null-portable-v2-smoke.json')
        assert small['status'] == 'standalone_exact_replay_passed'
        assert small['source_sha256'] == sha(Path(__file__))
        assert small['paired_proof_sha256'] == sha(proof_path)
    outputs = ROOT / '.biohub/cache/trajectory-event-anchor-production-v1-output'
    harvest_path = ROOT / 'reports/experiments/trajectory-event-anchor-production-v1-harvest.json'
    assert sha(harvest_path) == proof['public_cache_harvest_sha256']
    harvest = read(harvest_path)
    model_path = ROOT / '.biohub/cache/trajectory-event-fork16-learned-44b6-portable-v1/event-trajectory-model.json'
    assert sha(model_path) == proof['model_sha256']
    model = read(model_path)
    target.mkdir()
    module = target / 'event-trajectory.py'
    module.write_bytes(code.encode('utf-8'))
    if not smoke:
        assert sha(module) == small['code_sha256']
    runtime = runpy.run_path(str(module))
    result = dict(status='replaying_standalone', source_sha256=sha(Path(__file__)),
        code_sha256=sha(module), helper_sha256=helper_sha, paired_proof_sha256=sha(proof_path),
        model_sha256=sha(model_path), records={}, ground_truth_opened=False, gpu_used=False,
        staged_worker_changed=False, live_jobs_changed=False, authorized_for_submission=False,
        kaggle_runtime_accepted=False, score_improvement_not_established=True)

    def persist():
        result['seconds'] = time.monotonic() - started
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        for stem in (proof['movies'][:1] if smoke else proof['movies']):
            folders = list((outputs / 'trajectory-complete').glob('shard-*/' + stem + '-original'))
            assert len(folders) == 1
            folder = folders[0]
            for filename in ('pre-postprocess.json', 'structured-repaired-prediction.json', 'raw-candidates.npz'):
                path = folder / filename
                assert sha(path) == harvest['files'][path.relative_to(outputs).as_posix()]['sha256']
            initial = read(folder / 'pre-postprocess.json')
            baseline = read(folder / 'structured-repaired-prediction.json')
            with np.load(folder / 'raw-candidates.npz', allow_pickle=False) as raw:
                graph, details = runtime['refine'](initial, baseline, raw['coords'], raw['edges'], model['weights'],
                    per_frame_seconds=10., max_seconds=600.)
            frames = proof['records'][stem]['frames']
            validate_graph(graph, frames)
            check_event_stage(initial, baseline, graph, details, frames=frames)
            assert not details['budget_exhausted'] and not details['solver_fallbacks']
            reference = proof['records'][stem]['arms']['null_pruning']
            reference_path = ROOT / '.biohub/cache/trajectory-event-public-cache-v2-full' / (stem + '-null_pruning.json')
            assert sha(reference_path) == reference['prediction_sha256']
            assert graph == read(reference_path)
            assert {k: v for k, v in details.items() if k != 'seconds'} == {
                k: v for k, v in reference['details'].items() if k != 'seconds'}
            result['records'][stem] = dict(exact_graph=True, exact_solver_records=True,
                processed_frames=details['processed_frames'], seconds=details['seconds'])
            persist()
            print(json.dumps(dict(stem=stem, **result['records'][stem])), flush=True)
        result['status'] = 'standalone_exact_replay_passed'
        persist()
    except BaseException as error:
        result.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=1, user_api='blas'):
        main(parser.parse_args().smoke)
