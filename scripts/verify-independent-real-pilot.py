"""Verify downloaded pilot against its launch notebook, never against GT."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embedded_sources(notebook):
    nb = json.loads(notebook.read_text())
    found = {}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ('sources', 'runtime_sources'):
                    if target.id in found:
                        raise ValueError('Duplicate embedded source bundle')
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != {'sources', 'runtime_sources'}:
        raise ValueError('Missing launch source bundles')
    return found


def check_history(history, steps, maximum_steps=100):
    if not history or [r['step'] for r in history] != list(range(10, steps+1, 10)):
        raise ValueError('Missing or inconsistent checkpoint history')
    if maximum_steps not in (100,1000) or steps < 10 or steps > maximum_steps or steps % 10:
        raise ValueError('Invalid pilot step count')
    elapsed = 0.
    for row in history:
        for field in ('edge_loss', 'detection_loss', 'elapsed_seconds'):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError('Nonfinite or negative training telemetry')
        if row['elapsed_seconds'] <= elapsed:
            raise ValueError('Nonmonotonic training clock')
        elapsed = row['elapsed_seconds']
        if not 0 <= row['max_detected_nodes'] <= 2048:
            raise ValueError('Detection guard violated')
        if len(row['gpu_peak_allocated']) != 2 or any(v <= 0 for v in row['gpu_peak_allocated']):
            raise ValueError('Missing two-GPU allocation evidence')


def verify(root, notebook):
    expected = embedded_sources(notebook)
    for name, folder, manifest_name in (
        ('sources', 'repo', 'source_hashes.json'),
        ('runtime_sources', 'runtime', 'runtime_hashes.json')):
        hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p,s in expected[name].items()}
        if json.loads((root/manifest_name).read_text()) != hashes:
            raise ValueError('Launch source manifest mismatch')
        for path, digest in hashes.items():
            target = (root/folder/path).resolve()
            if not target.is_relative_to((root/folder).resolve()) or sha(target) != digest:
                raise ValueError('Downloaded source hash mismatch')
    terminal = json.loads((root/'launcher_terminal.json').read_text())
    result = json.loads((root/'outputs/result.json').read_text())
    identity = json.loads((root/'outputs/identity.json').read_text())
    split = json.loads(expected['runtime_sources']['split.json'])
    if terminal['status'] != 'completed' or result['status'] != 'completed_pilot':
        raise ValueError('Successful pilot terminal required; partial checkpoints need separate recovery')
    if terminal['run_id'] != result['run_id'] or result['run_id'] != 'independent-real-pilot-v1':
        raise ValueError('Run identity mismatch')
    if terminal['submission_performed'] is not False or terminal['declared_budget_seconds'] != 3600:
        raise ValueError('Launcher scope violation')
    if not 0 < terminal['elapsed_seconds'] <= 3600:
        raise ValueError('Runtime budget violation')
    for key, value in identity.items():
        if result.get(key) != value:
            raise ValueError('Result identity mismatch')
    if identity['training_stems'] != split['folds'][0]['train'][:4]:
        raise ValueError('Training movie scope violation')
    if identity.get('held_out_embryo') != split['folds'][0]['held_out_embryo'] or identity.get('seed') != 20260909:
        raise ValueError('Training fold/seed identity mismatch')
    if 'detector_warmup_steps=50' in expected['runtime_sources']['run_pilot.py']:
        if identity.get('detector_warmup_steps') != 50 or identity.get('detector_initial_probability') != .01:
            raise ValueError('Launch warm-up configuration mismatch')
    if 'empty_attention_guard=True' in expected['runtime_sources']['run_pilot.py'] and identity.get('empty_attention_guard') is not True:
        raise ValueError('Missing launched attention guard identity')
    if identity['manifest_sha256'] != hashlib.sha256(expected['runtime_sources']['split.json'].encode()).hexdigest():
        raise ValueError('Training split hash mismatch')
    for key in ('target_audit_opened', 'public_checkpoint_loaded', 'authorized_for_submission',
                'selection_opened', 'production_promotion_authorized'):
        if result[key] is not False:
            raise ValueError('Pilot training/authorization scope violation')
    history = result['history']
    if json.loads((root/'outputs/history.json').read_text()) != history:
        raise ValueError('Training history file mismatch')
    maximum_steps = json.loads(notebook.read_text())['metadata']['codex'].get('max_steps',100)
    if identity.get('max_steps',100) != maximum_steps:
        raise ValueError('Training duration differs from launch profile')
    check_history(history, result['trained_steps'], maximum_steps)
    if result['trained_steps'] > 100:
        smoke = result.get('gpu_smoke')
        if not isinstance(smoke, dict):
            raise ValueError('Missing required GPU functionality gate')
        if json.loads((root/'outputs/gpu_smoke.json').read_text()) != smoke:
            raise ValueError('GPU smoke receipt mismatch')
        if (smoke['status'] != 'passed' or smoke['device'] != 'cuda'
                or smoke['checkpoint_step'] != 100 or smoke['frames'] != 3
                or smoke['movie'] not in identity['training_stems']
                or smoke['strict_reload'] is not True or smoke['geff_round_trip'] is not True):
            raise ValueError('Missing required GPU functionality gate')
        if sha(root/'outputs/smoke_checkpoint.pt') != smoke['checkpoint_sha256']:
            raise ValueError('GPU smoke checkpoint hash mismatch')
        counts = smoke.get('scorer_counts', {})
        if not {'edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'} <= set(counts):
            raise ValueError('Missing organizer scorer smoke evidence')
    if identity.get('empty_attention_guard'):
        previous_updates = 0
        for row in history:
            matched, annotated = row['matched_training_nodes'], row['annotated_training_nodes']
            if not 0 <= matched <= annotated or annotated <= 0:
                raise ValueError('Invalid training coverage counts')
            if not math.isclose(row['training_node_recall'], matched/annotated, abs_tol=1e-12):
                raise ValueError('Training recall telemetry mismatch')
            if not 0 <= row['supervised_training_pairs'] <= 20:
                raise ValueError('Invalid supervised pair count')
            if not previous_updates <= row['optimizer_steps_max'] <= row['step']:
                raise ValueError('Invalid optimizer update telemetry')
            previous_updates = row['optimizer_steps_max']
            if not math.isfinite(row['grad_scaler_scale']) or row['grad_scaler_scale'] <= 0:
                raise ValueError('Invalid gradient scale')
    checkpoint = root/'outputs/last.pt'
    if sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if result['trained_steps'] > 100:
        smoke_state = torch.load(root/'outputs/smoke_checkpoint.pt', map_location='cpu', weights_only=True)
        if smoke_state['step'] != 100 or smoke_state['identity'] != identity:
            raise ValueError('GPU smoke checkpoint identity mismatch')
    if state['step'] != result['trained_steps'] or state['identity'] != identity:
        raise ValueError('Checkpoint identity mismatch')
    if not state['model'] or not all(torch.isfinite(v).all().item() for v in state['model'].values()):
        raise ValueError('Empty or nonfinite model')
    if identity.get('detector_warmup_steps'):
        warmup = result['warmup_history']
        if ([r['step'] for r in warmup] != [10,20,30,40,50]
                or state.get('warmup_steps') != 50
                or any(not math.isfinite(r['loss']) for r in warmup)):
            raise ValueError('Incomplete or nonfinite detector warm-up')
        if json.loads((root/'outputs/warmup_history.json').read_text()) != warmup:
            raise ValueError('Warm-up history file mismatch')
    for key in ('optimizer', 'scaler', 'torch_rng', 'cuda_rng', 'loader_rng', 'python_rng', 'numpy_rng'):
        if key not in state:
            raise ValueError('Incomplete resumable state')
    if len(state['cuda_rng']) != 2:
        raise ValueError('Missing two-device RNG state')
    optimizer_steps = [int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    if not optimizer_steps or min(optimizer_steps) < 1 or max(optimizer_steps) > result['trained_steps']:
        raise ValueError('No valid optimizer update evidence')
    return dict(status='verified_training_pilot', trained_steps=result['trained_steps'],
        optimizer_steps_min=min(optimizer_steps), optimizer_steps_max=max(optimizer_steps),
        first_block=history[0], last_block=history[-1],
        checkpoint_sha256=result['checkpoint_sha256'], notebook_sha256=sha(notebook),
        detection_loss_change=history[-1]['detection_loss']-history[0]['detection_loss'],
        edge_loss_change=history[-1]['edge_loss']-history[0]['edge_loss'],
        blocks_with_nonzero_association_loss=sum(r['edge_loss'] > 0 for r in history),
        final_block_max_detected_nodes=history[-1]['max_detected_nodes'],
        final_block_training_node_recall=history[-1].get('training_node_recall'),
        detector_coverage_established=False,
        validation_scope='Source-only optimization and throughput; no held-out evaluation',
        longer_gpu_run_authorized=False, authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--notebook', type=Path, default=ROOT/'kaggle/biohub-independent-real-pilot-v1/biohub-independent-real-pilot-v1.ipynb')
    args = parser.parse_args()
    print(json.dumps(verify(args.output_root, args.notebook), indent=2))
