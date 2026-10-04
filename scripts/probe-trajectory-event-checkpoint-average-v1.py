"""Small fixed source-only rare/ordinary exact-objective probe, not validation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_fast_case_store_v1 import CaseStore
from research.trajectory_event_vectorized_training_v1 import hinge

CASES = {'44b6': {'ordinary': [0, 1], 'division': [160, 258]},
         '6bba': {'ordinary': [0, 1], 'division': [66, 160]}}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(smoke):
    started = time.monotonic()
    reports = ROOT / 'reports/experiments'
    name = 'trajectory-event-checkpoint-average-v1-probe-' + ('smoke' if smoke else 'full')
    receipt, target = reports / (name + '.json'), ROOT / '.biohub/cache' / name
    assert not receipt.exists() and not target.exists(), 'Inspect existing probe'
    proof_path = reports / 'trajectory-event-checkpoint-average-v1.json'
    assert sha(proof_path) == '75dc1dcf4253dd2e964dc4acccbbbb396124afa05cfa526677120e7a418d8e6d'
    proof = read(proof_path)
    assert proof['status'] == 'fixed_checkpoint_averages_constructed_not_validated'
    if not smoke:
        small = read(reports / 'trajectory-event-checkpoint-average-v1-probe-smoke.json')
        assert small['status'] == 'source_objective_probe_complete' and small['source_sha256'] == sha(Path(__file__))
    chosen = {'6bba': {kind: indices[:1] for kind, indices in CASES['6bba'].items()}} if smoke else CASES
    target.mkdir()
    report = dict(status='probing_fixed_source_cases', source_sha256=sha(Path(__file__)),
        average_proof_sha256=sha(proof_path), source_only=True, ground_truth_files_opened=False,
        source_labels_from_original_training_cases=True, selection_or_validation_opened=False,
        plan=chosen, source_case_selection='first_two_ordinary_and_first_two_forced_fork_cases_in_frozen_order',
        full_training_risk_not_estimated=True, authorized_for_submission=False, gpu_used=False,
        source_smoke_is_not_quality_validation=True, records={})

    def persist():
        report['seconds'] = time.monotonic() - started
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        for embryo, kinds in chosen.items():
            folder = ROOT / '.biohub/cache' / ('trajectory-event-fork16-fit-' + embryo + '-v1')
            contract_path = folder / 'CONTRACT.json'
            assert sha(contract_path) == proof['models'][embryo]['fit_contract_sha256']
            contract = read(contract_path)
            store = CaseStore(ROOT / '.biohub/cache', contract['case_records'], cache_size=2)
            final_path = folder / 'final-weights.npz'
            average_path = ROOT / '.biohub/cache/trajectory-event-checkpoint-average-v1' / (embryo + '-average.npz')
            assert sha(final_path) == proof['models'][embryo]['final_weights_sha256']
            assert sha(average_path) == proof['models'][embryo]['model_sha256']
            with np.load(final_path, allow_pickle=False) as data:
                weights = dict(anchor=data['anchor'].copy(), final=data['weights'].copy())
            with np.load(average_path, allow_pickle=False) as data:
                weights['average'] = data['weights'].copy()
            penalty, anchor = np.array(contract['regularization']), np.array(contract['anchor'])
            assert np.array_equal(weights['anchor'], anchor)
            report['records'][embryo] = []
            for kind, indices in kinds.items():
                for index in indices:
                    record = contract['case_records'][index]
                    path = ROOT / '.biohub/cache' / record['path']
                    assert sha(path) == record['sha256']
                    with np.load(path, allow_pickle=False) as raw:
                        targets = raw['targets']
                        _, counts = np.unique(targets[targets >= 0], return_counts=True)
                        forced = int(np.sum(counts == 2))
                    assert (forced > 0) == (kind == 'division')
                    case = store[index]
                    row = dict(kind=kind, index=index, stem=record['stem'], t=record['t'],
                        case_sha256=sha(path), forced_forks=forced, known_constraints=case['n_constraints'], arms={})
                    for arm, w in weights.items():
                        begin = time.monotonic()
                        loss, gradient = hinge(case, w, time_limit=10.)
                        delta = w - anchor
                        regularizer = float(.5 * (penalty * delta) @ delta)
                        row['arms'][arm] = dict(hinge=float(loss), regularizer=regularizer,
                            objective=float(loss + regularizer), gradient_norm=float(np.linalg.norm(gradient)),
                            seconds=time.monotonic() - begin)
                    report['records'][embryo].append(row)
                    persist()
                    print(json.dumps(dict(embryo=embryo, **row)), flush=True)
        report['by_source_and_kind'] = {}
        for embryo, records in report['records'].items():
            report['by_source_and_kind'][embryo] = {}
            for kind in ('ordinary', 'division'):
                selected = [r for r in records if r['kind'] == kind]
                report['by_source_and_kind'][embryo][kind] = {a: float(np.mean([r['arms'][a]['objective'] for r in selected])) for a in ('anchor', 'final', 'average')}
        report['status'] = 'source_objective_probe_complete'
        persist()
    except BaseException as error:
        report.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=1, user_api='blas'):
        main(parser.parse_args().smoke)
