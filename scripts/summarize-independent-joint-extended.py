"""Verify small/long real training receipts without host Torch imports."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def validate(result, terminal, split, steps):
    if steps not in (100, 6000):
        raise ValueError('Unknown training length')
    identity = result['identity']
    if (result['status'] != 'completed' or terminal['status'] != 'completed'
        or terminal['run_id'] != 'independent-joint-extended-v1'
        or not 0 < terminal['elapsed_seconds'] < terminal['declared_budget_seconds'] <= 7200
        or terminal['submission_performed'] is not False):
        raise ValueError('Completed bounded experiment required')
    if (identity['max_steps'] != steps or identity.get('checkpoint_bn_once') is not True
        or identity.get('joint_training') is not True or identity['training_profile'] != 'full'
        or identity['initialization_sha256'] != SOURCE_SHA
        or identity['loss'] != 'sparse_parent_classification_with_null_v1'
        or identity['training_stems'] != split['folds'][0]['train']
        or identity['selection_opened'] is not False or identity['target_audit_opened'] is not False
        or result['authorized_for_submission'] is not False):
        raise ValueError('Unexpected training provenance')
    if set(identity['training_stems']) & set(split['folds'][0]['selection'] + split['folds'][0]['audit_order']):
        raise ValueError('Training contamination')
    if len(result['sample_hashes']) != steps * 2:
        raise ValueError('Input count mismatch')
    history = result['history']
    if [r['step'] for r in history] != list(range(10, steps + 1, 10)):
        raise ValueError('Incomplete update coverage')
    for row in history:
        if (len(row['batchnorm_update_deltas']) != 10 or
            any(n != row['step'] for n in row['batchnorm_update_deltas'].values())):
            raise ValueError('Repeated or missing BatchNorm updates')
    if history[-1]['optimizer_steps_min'] < .9 * steps or history[-1]['optimizer_steps_max'] > steps:
        raise ValueError('Optimizer update gate failed')
    for name, value in result['initial_module_hashes'].items():
        if result['final_module_hashes'][name] == value:
            raise ValueError('A model module remained frozen')
    gates = [('before', 0), ('after', steps)] + ([('step100_smoke', 100)] if steps > 100 else [])
    for key, step in gates:
        gate = result[key]
        if (gate['status'] != 'passed' or gate['checkpoint_step'] != step
            or gate['strict_reload'] is not True or gate['geff_round_trip'] is not True
            or gate['movie'] not in identity['training_stems']):
            raise ValueError('Missing real reload/inference gate')
    if result['after']['checkpoint_sha256'] != result['checkpoint_sha256']:
        raise ValueError('Final checkpoint hash mismatch')
    return dict(status='verified_training_receipts', identity=identity,
        elapsed_seconds=terminal['elapsed_seconds'], checkpoint_sha256=result['checkpoint_sha256'],
        input_samples=len(result['sample_hashes']), first_block=history[0], last_block=history[-1],
        before=result['before'], after=result['after'], step100_smoke=result['step100_smoke'],
        initial_module_hashes=result['initial_module_hashes'], final_module_hashes=result['final_module_hashes'],
        scope='Training functionality only; no complete-movie accuracy or submission authorization',
        authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--steps', type=int, choices=(100,6000), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.root / 'outputs/result.json'
    report = validate(json.loads(path.read_text()), json.loads((args.root / 'launcher_terminal.json').read_text()),
                      json.loads((ROOT / 'research/independent_real_baseline_v1_split.json').read_text()), args.steps)
    report['full_result_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
