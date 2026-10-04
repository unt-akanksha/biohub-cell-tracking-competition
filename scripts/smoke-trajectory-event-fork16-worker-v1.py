"""Replay the actual drafted worker call on a real movie, without GPU or labels."""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_event_release_v1 import check_event_stage


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(embryo):
    started = time.monotonic()
    name = 'trajectory-event-fork16-worker-' + embryo + '-v1'
    stage = ROOT / '.biohub/staging' / name
    draft = read(stage / 'DRAFT.json')
    assert draft['status'] == 'local_worker_overlay_draft_only' and not draft['authorized_for_submission']
    assert sha(stage / 'overlays.json') == draft['overlays_sha256']
    overlays = read(stage / 'overlays.json')
    model = json.loads(overlays['event-trajectory-model.json'])
    assert model['max_fork_children'] == 16 and not model['authorized_for_submission']
    portable_name = 'trajectory-event-fork16-learned-' + embryo + '-portable-v1'
    proof_path = ROOT / 'reports/experiments' / (portable_name + '.json')
    assert sha(proof_path) == draft['portable_smoke_sha256']
    proof = read(proof_path)
    assert proof['status'] == 'learned_portable_smoke_exact'
    stem = proof['movies'][0]  # Previously fixed small smoke, no score-based choice.
    source_model_path = ROOT / '.biohub/cache' / portable_name / 'event-trajectory-model.json'
    assert sha(source_model_path) == proof['model_sha256'] and model == read(source_model_path)
    reference = ROOT / '.biohub/cache' / portable_name / (stem + '-prediction.json')
    assert sha(reference) == proof['inference'][stem]['prediction_sha256']
    held = '6bba' if embryo == '44b6' else '44b6'
    original = read(ROOT / 'reports/experiments' / ('trajectory-event-fork16-held-' + held + '-v1-smoke.json'))
    assert original['model_sha256'] == model['weights_sha256']
    located = []
    for batch in range(8):
        prefix = 'trajectory-event-source-v1-b' + str(batch)
        features = ROOT / '.biohub/cache' / (prefix + '-features')
        frozen = read(features / 'RESULT.json')
        if stem not in frozen['per_movie']:
            continue
        assert sha(features / 'RESULT.json') == original['source_scope_receipts'][str(batch)]['feature_receipt_sha256']
        base_path = features / (stem + '-prediction.json')
        assert sha(base_path) == frozen['per_movie'][stem]['prediction_sha256']
        backup_path = ROOT / 'reports/experiments' / (prefix + '-full-harvest.json')
        assert sha(backup_path) == original['source_scope_receipts'][str(batch)]['backup_sha256']
        backup = {r['path']: r['sha256'] for r in read(backup_path)['records']}
        origin = ROOT / '.biohub/cache' / (prefix + '-full-output') / (stem + '-original')
        for filename in ('pre-postprocess.json', 'raw-candidates.npz'):
            assert sha(origin / filename) == backup[stem + '-original/' + filename]
        with np.load(origin / 'raw-candidates.npz', allow_pickle=False) as raw:
            located.append((read(origin / 'pre-postprocess.json'), read(base_path), raw['coords'].copy(), raw['edges'].copy()))
    assert len(located) == 1
    initial, baseline, coords, edges = located[0]
    runtime = dict(__name__='worker_event_replay')
    exec(compile(overlays['event-trajectory.py'], '<draft-event>', 'exec'), runtime)
    assert tuple(model['features']) == runtime['FEATURES']
    tree = ast.parse(overlays['portable-worker.py'])
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'event' and n.func.attr == 'refine']
    assert len(calls) == 1
    limits = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    assert limits == dict(per_frame_seconds=10., max_seconds=600.)
    target = ROOT / '.biohub/cache' / (name + '-smoke')
    receipt = ROOT / 'reports/experiments' / (name + '-smoke.json')
    assert not target.exists() and not receipt.exists()
    target.mkdir()
    result = dict(status='running_actual_worker_call', stem=stem, fit_embryo=embryo,
        source_sha256=sha(Path(__file__)), draft_sha256=sha(stage / 'DRAFT.json'),
        model_values_exactly_preserved=True, source_model_json_sha256=sha(source_model_path),
        source_json_line_endings_may_be_normalized_in_overlay=True, inference_limits=limits,
        ground_truth_opened=False, gpu_used=False, full_gpu_worker_executed=False,
        notebook_created=False, authorized_for_submission=False)

    def persist():
        result['seconds'] = time.monotonic() - started
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        environment = dict(event=SimpleNamespace(refine=runtime['refine']), raw_ilp=initial,
            repaired=baseline, coords=coords, edges=edges, np=np, event_model=model)
        candidate, details = eval(compile(ast.Expression(calls[0]), '<actual-worker-event-call>', 'eval'), environment)
        validate_graph(candidate, 100)
        invariants = check_event_stage(initial, baseline, candidate, details, frames=100)
        output = target / (stem + '-prediction.json')
        output.write_text(json.dumps(candidate, sort_keys=True, allow_nan=False), encoding='utf-8')
        result.update(details=details, invariants=invariants, prediction_sha256=sha(output))
        persist()
        assert not details['budget_exhausted'] and not details['solver_fallbacks']
        assert candidate == read(reference)
        assert {k: v for k, v in details.items() if k != 'seconds'} == {
            k: v for k, v in proof['inference'][stem]['details'].items() if k != 'seconds'}
        result.update(status='actual_worker_event_call_replay_exact', exact_graph_and_solver_records=True)
        persist()
        print(json.dumps(dict(status=result['status'], stem=stem, inference_seconds=details['seconds'], seconds=result['seconds'])))
    except BaseException as error:
        result.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-embryo', choices=('44b6', '6bba'), required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        main(parser.parse_args().fit_embryo)
