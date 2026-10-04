"""Local paired runtime diagnostic on cached public-test network outputs, no GT."""
import argparse
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
from research.trajectory_event_vectorized_inference_v1 import private_function
from research.trajectory_event_null_dominance_v1 import allowed_options


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(smoke):
    started = time.monotonic()
    name = 'trajectory-event-public-cache-v2-' + ('smoke' if smoke else 'full')
    reports = ROOT / 'reports/experiments'
    receipt = reports / (name + '.json')
    target = ROOT / '.biohub/cache' / name
    assert not receipt.exists() and not target.exists()
    proof_path = reports / 'trajectory-event-null-portable-v1-full.json'
    assert sha(proof_path) == '756222160dc4d8e825c0e68292c5b8e8da58523550ef319eca63a423e8a34b3b'
    proof = read(proof_path)
    assert proof['status'] == 'complete_portable_graphs_exact' and len(proof['records']) == 2
    assert proof['helper_sha256'] == sha(ROOT / 'research/trajectory_event_null_dominance_v1.py')
    if not smoke:
        small = read(reports / 'trajectory-event-public-cache-v2-smoke.json')
        assert small['status'] == 'paired_public_runtime_verified' and small['source_sha256'] == sha(Path(__file__))
    portable = ROOT / '.biohub/cache/trajectory-event-fork16-vectorized-portable-v1/event-trajectory.py'
    assert sha(portable) == 'eac9ee4f7ed8b5192167e94a39910e8343ff1e1f90880f95dd54360ef9abc161'
    model_path = ROOT / '.biohub/cache/trajectory-event-fork16-learned-44b6-portable-v1/event-trajectory-model.json'
    assert sha(model_path) == proof['model_sha256']
    model = read(model_path)
    runtime = runpy.run_path(str(portable))
    optimized_infer = private_function(runtime['infer'], {'allowed_options': allowed_options})
    optimized = private_function(runtime['refine_prepared'], {'infer': optimized_infer})
    outputs = ROOT / '.biohub/cache/trajectory-event-anchor-production-v1-output'
    harvest_path = reports / 'trajectory-event-anchor-production-v1-harvest.json'
    assert sha(harvest_path) == '6959398146a025cae14cadc02a0a32c9874dabc2525e6584ee8b3c43d78d3f77'
    harvest = read(harvest_path)
    terminal_path = outputs / 'trajectory-complete/result.json'
    assert sha(terminal_path) == harvest['files']['trajectory-complete/result.json']['sha256']
    terminal = read(terminal_path)
    assert terminal['status'] == 'complete' and terminal['mode'] == 'production'
    movies = []
    for shard, worker in terminal['workers'].items():
        assert not worker['ground_truth_opened'] and worker['inputs_unchanged']
        for stem, record in worker['movies'].items():
            folder = outputs / 'trajectory-complete' / ('shard-' + shard) / (stem + '-original')
            movies.append((int(record['original']['nodes']), stem, folder, int(record['original']['frames'])))
    assert len(movies) == 4 and len({r[1] for r in movies}) == 4
    chosen = sorted(movies)[:1] if smoke else sorted(movies)
    target.mkdir()
    result = dict(status='public_cache_pair_running', source_sha256=sha(Path(__file__)),
        model_sha256=sha(model_path), runtime_sha256=sha(portable), source_replay_sha256=sha(proof_path),
        public_cache_harvest_sha256=sha(harvest_path), movies=[r[1] for r in chosen], records={},
        source44_model_quality_not_assumed=True, ground_truth_opened=False, gpu_used=False,
        network_outputs_reused_for_local_diagnostic_only=True, not_a_submission=True,
        live_jobs_or_staging_changed=False, kaggle_runtime_accepted=False,
        authorized_for_submission=False, local_times_not_direct_kaggle_forecast=True)

    def persist():
        result['seconds'] = time.monotonic() - started
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        for index, (_, stem, folder, frames) in enumerate(chosen):
            for filename in ('pre-postprocess.json', 'structured-repaired-prediction.json', 'raw-candidates.npz', 'repair-details.json'):
                path = folder / filename
                assert sha(path) == harvest['files'][path.relative_to(outputs).as_posix()]['sha256']
            initial, baseline = read(folder / 'pre-postprocess.json'), read(folder / 'structured-repaired-prediction.json')
            begin = time.perf_counter()
            with np.load(folder / 'raw-candidates.npz', allow_pickle=False) as raw:
                coords, edges = raw['coords'], raw['edges']
                for ident, node in initial['nodes'].items():
                    assert np.array_equal(coords[int(ident)], [node[k] for k in ('t', 'z', 'y', 'x')])
                groups = runtime['candidates'](initial, baseline)
                features = runtime['edge_features'](initial, baseline, groups, edges)
            row = dict(frames=frames, feature_seconds=time.perf_counter() - begin,
                       actual_previous_kaggle_event_details=read(folder / 'repair-details.json')['event_assignment'], arms={})
            result['records'][stem] = row
            order = [('reference', runtime['refine_prepared']), ('null_pruning', optimized)]
            if index % 2:
                order.reverse()
            row['arm_order'] = [arm for arm, _ in order]
            graphs = {}
            for arm, infer in order:
                graph, details = infer(initial, baseline, groups, features, model['weights'], per_frame_seconds=10., max_seconds=600.)
                validate_graph(graph, frames)
                check_event_stage(initial, baseline, graph, details, frames=frames)
                output = target / (stem + '-' + arm + '.json')
                output.write_text(json.dumps(graph, sort_keys=True, allow_nan=False), encoding='utf-8')
                row['arms'][arm] = dict(details=details, prediction_sha256=sha(output))
                graphs[arm] = graph
                persist()
                assert not details['solver_fallbacks'] and not details['budget_exhausted'], (stem, arm, 'incomplete exact inference')
            assert graphs['reference'] == graphs['null_pruning'], 'Public-test graph changed'
            row['exact_paired_graph'] = True
            persist()
            print(json.dumps(dict(stem=stem, feature_seconds=row['feature_seconds'],
                reference_seconds=row['arms']['reference']['details']['seconds'],
                pruned_seconds=row['arms']['null_pruning']['details']['seconds'], exact_paired_graph=True)), flush=True)
        result['status'] = 'paired_public_runtime_verified'
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
