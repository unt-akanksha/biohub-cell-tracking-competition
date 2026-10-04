"""Archive exact real-data probe/fit evidence before any full selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INITIAL = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def verify(result, terminal, split, steps, probe=None, *, initial=INITIAL,
           loss='sparse_parent_with_annotated_missing_parent_null_v1'):
    identity = result['identity']
    if (result['status'] != 'completed' or terminal['status'] != 'completed'
            or not 0 < terminal['elapsed_seconds'] <= 3600
            or identity['initialization_sha256'] != initial or identity['max_steps'] != steps
            or identity['training_stems'] != split['folds'][0]['train']
            or identity['training_profile'] != 'full'
            or identity['loss'] != loss
            or identity['known_null'] != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False)
            or identity['frozen_modules'] != ['unet','detect_head']
            or result['detector_unchanged'] is not True
            or result['initial_detector_sha256'] != result['frozen_detector_sha256']
            or any(identity[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))
            or result['authorized_for_submission'] is not False or terminal['submission_performed'] is not False):
        raise ValueError('Invalid completed frozen-detector known-null evidence')
    history = result['history']
    if [r['step'] for r in history] != list(range(10,steps+1,10)):
        raise ValueError('Incomplete optimization history')
    if any(not math.isfinite(r['parent_loss']) or r['max_nodes'] > 2048 for r in history):
        raise ValueError('Nonfinite loss or exceeded candidate bound')
    total = sum(r['known_null_columns'] for r in history)
    if total <= 0 or result['known_null_supervised_total'] != total or len(result['sample_hashes']) != 2*steps:
        raise ValueError('Missing real null supervision or exact input coverage')
    for name, step in [('before',0),('after',steps)]+([('step100_smoke',100)] if steps == 1000 else []):
        smoke = result[name]
        if (smoke['status'] != 'passed' or smoke['checkpoint_step'] != step
                or not smoke['strict_reload'] or not smoke['geff_round_trip']
                or smoke['predicted_nodes'] != result['before']['predicted_nodes']):
            raise ValueError('Missing real reload/node-preservation gate')
    if result['after']['checkpoint_sha256'] != result['checkpoint_sha256']:
        raise ValueError('Final checkpoint was not tested')
    if probe is not None and result['sample_hashes'][:200] != probe['sample_hashes']:
        raise ValueError('First100 paired augmented training inputs changed')
    return dict(status='verified_training_receipts_not_selection',steps=steps,
        checkpoint_sha256=result['checkpoint_sha256'],known_null_columns=total,
        positive_links=sum(r['positive_links'] for r in history),
        confident_null_columns=sum(r['confident_null_columns'] for r in history),
        detector_unchanged=True,launcher_seconds=terminal['elapsed_seconds'],
        optimization_seconds=history[-1]['elapsed_seconds'],before=result['before'],after=result['after'],
        probe_inputs_replayed=probe is not None,authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=int,choices=(1,2),required=True)
    args = parser.parse_args()
    cache = ROOT/'.biohub/cache/kernel-outputs'/('independent-known-null-probe-v1' if args.version == 1 else 'independent-known-null-fit-v2')/'independent_known_null'
    result_path = cache/'outputs/result.json'
    terminal_path = cache/'launcher_terminal.json'
    result = json.loads(result_path.read_text())
    probe_path = ROOT/'.biohub/cache/kernel-outputs/independent-known-null-probe-v1/independent_known_null/outputs/result.json'
    report = verify(result,json.loads(terminal_path.read_text()),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()),
        100 if args.version == 1 else 1000, None if args.version == 1 else json.loads(probe_path.read_text()))
    report['source_sha256'] = {name:hashlib.sha256(path.read_bytes()).hexdigest()
                              for name,path in [('result',result_path),('terminal',terminal_path)]}
    (ROOT/f'reports/experiments/independent-known-null-v{args.version}-training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','source_sha256')},indent=2))
