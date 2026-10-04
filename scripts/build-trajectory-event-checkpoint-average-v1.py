"""Recover fixed third-epoch averages; never choose snapshots by metric outcomes."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_checkpoint_average_v1 import snapshot_steps, average_snapshots
from research.trajectory_event_adam_state_v1 import update


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    name = 'trajectory-event-checkpoint-average-v1'
    target, reports = ROOT / '.biohub/cache' / name, ROOT / 'reports/experiments'
    receipt = reports / (name + '.json')
    assert not target.exists() and not receipt.exists(), 'Preserve existing average experiment'
    expected_reports = dict(
        e44='db22184988a00299c0f5827ec860277831705efeb363fa74af72429e46582c31',
        e6='9bb0ef61d09c13fa871f6f9b61870d837aea5a798b534e0a019a9112a53a62cd')
    report = dict(status='constructing_fixed_averages', source_sha256=sha(Path(__file__)),
        design_sha256=sha(reports / (name + '-design.md')),
        helper_sha256=sha(ROOT / 'research/trajectory_event_checkpoint_average_v1.py'),
        source_only=True, new_ground_truth_opened=False, metric_based_snapshot_selection=False,
        previous_failures_preserved=True, prior_source_and_selection_exposure_disclosed=True,
        training_restarted=False, gpu_used=False, authorized_for_submission=False, models={})
    target.mkdir()

    def persist():
        report['seconds'] = time.monotonic() - started
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        for embryo, key, tail_start in (('44b6', 'e44', 3400), ('6bba', 'e6', 13400)):
            folder = ROOT / '.biohub/cache' / ('trajectory-event-fork16-fit-' + embryo + '-v1')
            fit_path = reports / ('trajectory-event-fork16-fit-' + embryo + '-v1.json')
            assert sha(fit_path) == expected_reports[key]
            fit, contract = read(fit_path), read(folder / 'CONTRACT.json')
            assert fit['status'] == 'source_event_training_complete'
            assert sha(folder / 'CONTRACT.json') == fit['contract_sha256']
            for helper, digest in contract['helper_sha256'].items():
                assert sha(ROOT / 'research' / helper) == digest
            assert fit['steps'] == contract['cases'] * contract['epochs'] and contract['epochs'] == 3
            snapshots, records, trajectory = {}, [], []
            wanted = snapshot_steps(contract['cases'])
            for step in wanted:
                paths = list(folder.glob('checkpoint-' + str(step).zfill(7) + '-*.json'))
                assert len(paths) == 1
                path = paths[0]
                value = read(path)
                assert value['contract_sha256'] == fit['contract_sha256']
                assert value['optimizer']['steps'] == step
                weights = np.asarray(value['optimizer']['weights'], np.float64)
                assert weights.shape == (30,) and np.isfinite(weights).all()
                snapshots[step] = weights
                records.append(dict(step=step, path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
                trajectory.append(dict(step=step, geometry_norm=float(np.linalg.norm(weights[22:28]))))
            mean = average_snapshots(snapshots, contract['cases'])
            final_path = folder / 'final-weights.npz'
            assert sha(final_path) == fit['weights_sha256']
            with np.load(final_path, allow_pickle=False) as final:
                actual = final['weights'].copy()
                feature_names = final['feature_names'].copy()
                assert int(final['max_fork_children']) == 16
            tail_path = next(folder.glob('checkpoint-' + str(tail_start).zfill(7) + '-*.json'))
            tail = read(tail_path)
            assert tail['contract_sha256'] == fit['contract_sha256'] and tail['optimizer']['steps'] == tail_start
            state = {k: np.array(v, dtype=float) if k in ('weights', 'mean', 'variance') else v for k, v in tail['optimizer'].items()}
            penalty, anchor = np.array(contract['regularization']), np.array(contract['anchor'])
            for _ in range(tail_start, fit['steps']):
                update(state, penalty * (state['weights'] - anchor), contract['learning_rate'])
            model_path = target / (embryo + '-average.npz')
            np.savez_compressed(model_path, weights=mean, feature_names=feature_names, max_fork_children=16)
            with np.load(model_path, allow_pickle=False) as saved:
                assert np.array_equal(saved['weights'], mean)
            report['models'][embryo] = dict(snapshots=records, trajectory=trajectory,
                snapshot_count=len(records), model_sha256=sha(model_path), final_weights_sha256=sha(final_path),
                fit_contract_sha256=fit['contract_sha256'], fit_report_sha256=sha(fit_path),
                geometry_norm_final=float(np.linalg.norm(actual[22:28])), geometry_norm_average=float(np.linalg.norm(mean[22:28])),
                weights=mean.tolist(), tail_regularization_only_control=dict(
                    initial_checkpoint_sha256=sha(tail_path), start=tail_start, end=fit['steps'],
                    geometry_max_abs_discrepancy=float(np.max(np.abs(actual[22:28] - state['weights'][22:28])))))
            persist()
        report['status'] = 'fixed_checkpoint_averages_constructed_not_validated'
        persist()
        print(json.dumps(dict(status=report['status'], models={e: {k: v for k, v in r.items() if k not in ('trajectory', 'snapshots', 'weights')}
            for e, r in report['models'].items()})), flush=True)
    except BaseException as error:
        report.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    main()
