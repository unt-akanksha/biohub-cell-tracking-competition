"""Verify paired training receipts without importing host PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(root,split):
    terminal = json.loads((root/'launcher_terminal.json').read_text())
    result_path = root/'outputs/paired_result.json'
    pair = json.loads(result_path.read_text())
    if terminal['status'] != 'completed' or terminal['run_id'] != 'independent-motion-broad-pair-v1':
        raise ValueError('Completed broad-pair terminal required')
    if not 0 < terminal['elapsed_seconds'] < 3600 or terminal['submission_performed'] is not False:
        raise ValueError('Unexpected execution budget or submission')
    if pair['status'] != 'completed' or pair['inputs_identical'] is not True or pair['sample_count'] != 2000:
        raise ValueError('Incomplete paired experiment')
    results = pair['results']
    if set(results) != {'control','row'}:
        raise ValueError('Unexpected arms')
    if results['control']['sample_hashes'] != results['row']['sample_hashes']:
        raise ValueError('Actual input fingerprints differ')
    if results['control']['frozen_detector_sha256'] != results['row']['frozen_detector_sha256']:
        raise ValueError('Frozen detector fingerprints differ')
    for key in ('initialization_sha256','seed','optimizer','motion_residual','augmentation_seed'):
        if results['control']['identity'][key] != results['row']['identity'][key]:
            raise ValueError('Paired initialization or optimization contract differs')
    summary = {}
    for arm,result in results.items():
        identity = result['identity']
        expected_loss = ('sparse_parent_classification_with_null_v1' if arm == 'control'
                         else 'sparse_parent_and_row_hard_negative_v1')
        if identity['loss'] != expected_loss:
            raise ValueError('Unexpected arm objective')
        if (result['status'] != 'completed' or result['detector_unchanged'] is not True
            or identity['training_profile'] != 'full' or identity['training_stems'] != split['folds'][0]['train']
            or identity['max_steps'] != 1000 or len(result['sample_hashes']) != 2000):
            raise ValueError('Unexpected completed training scope')
        if set(identity['training_stems']) & set(split['folds'][0]['selection']+split['folds'][0]['audit_order']):
            raise ValueError('Evaluation contamination')
        history = result['history']
        if [row['step'] for row in history] != list(range(10,1001,10)):
            raise ValueError('Incomplete update coverage')
        for key,step in (('before',0),('step100_smoke',100),('after',1000)):
            gate = result[key]
            if (gate['status'] != 'passed' or gate['checkpoint_step'] != step
                or gate['strict_reload'] is not True or gate['geff_round_trip'] is not True):
                raise ValueError('Missing real GPU functionality gate')
        if result['after']['checkpoint_sha256'] != result['checkpoint_sha256']:
            raise ValueError('Final checkpoint and reload hashes differ')
        summary[arm] = dict(checkpoint_sha256=result['checkpoint_sha256'],
            loss=identity['loss'],training_totals={key:sum(row[key] for row in history)
                for key in ('correct','positive_links','confident_correct','wrong_child_claims')},
            first_block=history[0],last_block=history[-1],
            before_smoke_counts=result['before']['scorer_counts'],
            after_smoke_counts=result['after']['scorer_counts'])
    return dict(status='verified_training_receipts',elapsed_seconds=terminal['elapsed_seconds'],
        input_samples=2000,inputs_identical=True,detector_unchanged=True,arms=summary,
        full_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        scope='Training evidence only; complete-movie and embryo-held-out validation still required',
        authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    result = summarize(args.output_root,split)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
