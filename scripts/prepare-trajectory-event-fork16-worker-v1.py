"""Prepare local overlay draft only: no notebook, GPU launch, or model promotion."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_fork16_worker_v1 import adapt_worker


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(embryo):
    reports = ROOT / 'reports/experiments'
    name = 'trajectory-event-fork16-worker-' + embryo + '-v1'
    target = ROOT / '.biohub/staging' / name
    receipt = reports / (name + '.json')
    assert not target.exists() and not receipt.exists(), 'Preserve previous draft'
    previous = ROOT / '.biohub/staging/biohub-event-anchor-candidate-v2'
    build = read(reports / 'trajectory-event-anchor-final-v2-build.json')
    assert sha(previous / 'overlays.json') == build['overlay_sha256']
    assert sha(previous / 'derived-contract.json') == build['contract_sha256']
    parent = read(previous / 'derived-contract.json')
    overlays = read(previous / 'overlays.json')
    before = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in overlays.items()}
    assert all(parent['bundle_sha256'][key] == digest for key, digest in before.items())
    smoke_path = reports / ('trajectory-event-fork16-learned-' + embryo + '-portable-v1.json')
    smoke = read(smoke_path)
    assert smoke['status'] == 'learned_portable_smoke_exact' and smoke['fit_embryo'] == embryo
    assert not smoke['ground_truth_opened'] and not smoke['authorized_for_submission']
    assert len(smoke['inference']) == 2 and all(r['exact_original_graph_and_solver_records'] for r in smoke['inference'].values())
    code_path = ROOT / '.biohub/cache/trajectory-event-fork16-vectorized-portable-v1/event-trajectory.py'
    model_path = ROOT / '.biohub/cache' / ('trajectory-event-fork16-learned-' + embryo + '-portable-v1') / 'event-trajectory-model.json'
    assert sha(code_path) == smoke['code_sha256'] and sha(model_path) == smoke['model_sha256']
    model = read(model_path)
    assert model['max_fork_children'] == 16 and not model['authorized_for_submission']
    fit_path = reports / ('trajectory-event-fork16-fit-' + embryo + '-v1.json')
    fit = read(fit_path)
    assert fit['status'] == 'source_event_training_complete' and sha(fit_path) == model['fit_sha256']
    assert fit['weights_sha256'] == model['weights_sha256'] == smoke['weights_sha256']
    overlays['portable-worker.py'] = adapt_worker(overlays['portable-worker.py'])
    overlays['event-trajectory.py'] = code_path.read_text(encoding='utf-8')
    overlays['event-trajectory-model.json'] = model_path.read_text(encoding='utf-8')
    overlays['NOTICE.txt'] += '\nUNRELEASED learned fork16 overlay draft. Same licensed upstream image models; source-only learned event weights with explicit 10-second/frame and 600-second/movie event limits. Full quality, platform runtime, quota and release checks remain required. No GPU launch, Kaggle notebook or competition entry is supplied by this artifact.\n'
    after = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in overlays.items()}
    changed = sorted(key for key in before if before[key] != after[key])
    assert changed == ['NOTICE.txt', 'event-trajectory-model.json', 'event-trajectory.py', 'portable-worker.py']
    for key, value in overlays.items():
        if key.endswith('.py'):
            ast.parse(value)
    target.mkdir()
    output = target / 'overlays.json'
    output.write_text(json.dumps(overlays, indent=2), encoding='utf-8')
    result = dict(status='local_worker_overlay_draft_only', fit_embryo=embryo,
        source_sha256=sha(Path(__file__)), adapter_sha256=sha(ROOT / 'research/trajectory_event_fork16_worker_v1.py'),
        overlays_sha256=sha(output), overlay_file_sha256=after, changed_overlay_files=changed,
        parent_contract_sha256=build['contract_sha256'], parent_overlay_sha256=build['overlay_sha256'],
        fit_sha256=sha(fit_path), portable_smoke_sha256=sha(smoke_path),
        exact_model_weights_preserved=True, inference_deadlines_match_frozen_evaluator=True,
        event_per_frame_seconds=10., event_max_seconds=600., max_fork_children=16,
        full_quality_decision='pending_separate_evaluation', global_runtime_projection_required=True,
        notebook_created=False, platform_runtime_accepted=False, fresh_quota_check_required=True,
        required_kaggle_gpu_reserve_hours=8, gpu_launched=False, authorized_for_submission=False,
        input_parent_runtime_required=True, pending_submission_changed=False)
    text = json.dumps(result, indent=2) + '\n'
    (target / 'DRAFT.json').write_text(text, encoding='utf-8')
    receipt.write_text(text, encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-embryo', choices=('44b6', '6bba'), required=True)
    main(parser.parse_args().fit_embryo)
