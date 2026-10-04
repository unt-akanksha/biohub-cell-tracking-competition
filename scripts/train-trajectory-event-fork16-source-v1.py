"""Fixed source-embryo event fit with exact-order resumable checkpoints.

The opposite embryo is excluded from optimizer inputs. Existing public/backbone
and inherited edge-weight exposure remains disclosed; this is not pristine OOF.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_features_v1 import FEATURES
from research.trajectory_event_training_v1 import prior
from research.trajectory_event_fast_training_v1 import hinge
from research.trajectory_event_fast_case_store_v1 import CaseStore
from research.trajectory_event_adam_state_v1 import initialize, update


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def validate_schedule(state, progress, cases, epochs):
    epoch, position = progress['epoch'], progress['position']
    assert isinstance(epoch, int) and 0 <= epoch <= epochs
    assert isinstance(position, int) and 0 <= position < cases
    assert state['steps'] == epoch * cases + position
    assert len(progress['history']) == epoch
    assert np.isfinite(progress['loss_sum']) and progress['loss_sum'] >= 0
    if epoch < epochs:
        assert sorted(progress['order']) == list(range(cases)), 'Checkpoint order is not a full permutation'
    else:
        assert position == 0 and progress['order'] == []


@contextmanager
def exclusive_lock(path):
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embryo', choices=('44b6', '6bba'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--control-smoke', action='store_true')
    parser.add_argument('--steps-limit', type=int)
    parser.add_argument('--wall-seconds', type=int, default=7200)
    args = parser.parse_args()
    assert args.wall_seconds > 0 and (args.steps_limit is None or args.steps_limit > 0)
    assert not args.control_smoke or (args.steps_limit == 4 and not args.resume)
    reports = ROOT / 'reports/experiments'
    corpus_path = reports / 'trajectory-event-fork16-cases-controller-v1.json'
    corpus = read(corpus_path)
    assert corpus['status'] == 'complete_source_corpus_prepared'
    assert corpus['source_only'] and not corpus['selection_or_validation_opened']
    records, counts, stems, manifests = [], Counter(), [], {}
    for batch in range(8):
        name = 'trajectory-event-source-v1-b' + str(batch) + '-cases-fork16'
        path = reports / (name + '.json')
        assert sha(path) == corpus['completed'][str(batch)]['receipt_sha256']
        manifest = read(path)
        assert manifest['training_allowed'] and manifest['max_fork_children'] == 16
        for helper, expected in manifest['helper_sha256'].items():
            assert sha(ROOT / 'research' / helper) == expected
        manifests[name] = sha(path)
        for stem, item in manifest['per_movie'].items():
            if not stem.startswith(args.embryo + '_'):
                continue
            stems.append(stem)
            counts.update(item['source_prior_counts'])
            records.extend(dict(row, stem=stem, batch=batch, path=name + '/' + row['path'])
                           for row in item['cases'] if row['reason'] == 'prepared')
    assert len(set(stems)) == len(stems) == (13 if args.embryo == '44b6' else 47)
    initializer = ROOT / '.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
    assert sha(initializer) == 'e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
    with np.load(initializer, allow_pickle=False) as data:
        anchor = prior(data['6bba'], counts['annotated_division_parents'], counts['annotated_consecutive_parent_opportunities'])
    penalty = np.r_[np.full(18, .1), np.full(12, .02)]
    name = 'trajectory-event-fork16-fit-' + args.embryo + '-v1'
    if args.control_smoke:
        name += '-control-smoke'
    target, receipt = ROOT / '.biohub/cache' / name, reports / (name + '.json')
    contract = dict(source_embryo=args.embryo, source_stems=stems, cases=len(records), case_records=records,
        manifests=manifests, corpus_sha256=sha(corpus_path), source_prior_counts=dict(counts),
        anchor=anchor.tolist(), regularization=penalty.tolist(), feature_names=list(FEATURES),
        max_fork_children=16, epochs=3, learning_rate=.03, seed=20260914, oracle_seconds=10.,
        cache_limit_bytes=1536*1024**2, initializer_sha256=sha(initializer),
        source_only=True, selection_or_validation_used_for_training=False,
        opposite_embryo_excluded_from_optimizer=True, prior_backbone_and_edge_initializer_exposure=True,
        source_score_based_epoch_selection=False, source_sha256=sha(Path(__file__)),
        helper_sha256={n:sha(ROOT / 'research' / n) for n in ('trajectory_event_adam_state_v1.py',
            'trajectory_event_fast_training_v1.py', 'trajectory_event_fast_case_store_v1.py',
            'trajectory_event_training_v1.py', 'trajectory_event_assignment_v1.py', 'trajectory_event_dominance_v1.py')})
    if not args.resume:
        assert not target.exists() and not receipt.exists(), 'Previous fit exists; inspect it before --resume'
        target.mkdir()
        write_json(target / 'CONTRACT.json', contract)
    else:
        assert read(target / 'CONTRACT.json') == contract, 'Immutable training contract changed'
        assert read(receipt)['status'] in ('paused_at_step_limit', 'paused_at_wall_limit'), 'Resume only a verified paused fit'
    with exclusive_lock(target / 'training.lock'):
        started = time.monotonic()
        contract_sha = sha(target / 'CONTRACT.json')
        rng = np.random.default_rng(contract['seed'])
        if args.resume:
            pointer = read(target / 'latest-checkpoint.json')
            checkpoint = target / pointer['path']
            assert checkpoint.parent == target and sha(checkpoint) == pointer['sha256']
            saved = read(checkpoint)
            assert saved['contract_sha256'] == contract_sha
            state = {k:np.array(saved['optimizer'][k], dtype=np.float64) for k in ('weights', 'mean', 'variance')}
            state['steps'] = saved['optimizer']['steps']
            progress = saved['progress']
            rng.bit_generator.state = saved['rng_state']
        else:
            state = initialize(anchor)
            progress = dict(epoch=0, order=rng.permutation(len(records)).tolist(), position=0, loss_sum=0., history=[])
        validate_schedule(state, progress, len(records), contract['epochs'])
        invocation_start_step = state['steps']
        store = CaseStore(ROOT / '.biohub/cache', records, cache_size=len(records), max_cache_bytes=contract['cache_limit_bytes'])
        report = dict(status='running', pid=os.getpid(), source_only=True, source_embryo=args.embryo,
            contract_sha256=contract_sha, model_fitted=False, authorized_for_submission=False,
            selection_or_validation_opened=False, opposite_embryo_excluded_from_optimizer=True,
            resumed=args.resume, control_smoke=args.control_smoke,
            invocation_start_step=invocation_start_step, invocation_wall_cap=args.wall_seconds)
        def checkpoint():
            validate_schedule(state, progress, len(records), contract['epochs'])
            saved = dict(contract_sha256=contract_sha,
                optimizer={k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in state.items()},
                progress=progress, rng_state=rng.bit_generator.state)
            path = target / ('checkpoint-%07d-e%d-p%d.json' % (state['steps'], progress['epoch'], progress['position']))
            if path.exists():
                assert read(path) == saved, 'Conflicting checkpoint at the same optimization position'
            else:
                write_json(path, saved)
            write_json(target / 'latest-checkpoint.json', dict(path=path.name, sha256=sha(path)))
            report.update(steps=state['steps'], epochs_completed=progress['history'], checkpoint_sha256=sha(path),
                          seconds=time.monotonic()-started, cache_bytes=store.cache_bytes)
            write_json(receipt, report)
        checkpoint()
        try:
            # Hash-checked serialized cases before the first update of this invocation.
            for index in (0, len(records)-1):
                assert store[index]['x'].shape[1] == len(FEATURES)
            while progress['epoch'] < contract['epochs']:
                if time.monotonic()-started >= args.wall_seconds:
                    report['status'] = 'paused_at_wall_limit'
                    break
                if args.steps_limit is not None and state['steps']-invocation_start_step >= args.steps_limit:
                    report['status'] = 'paused_at_step_limit'
                    break
                index = progress['order'][progress['position']]
                case = store[index]
                assert case['x'].shape[1] == len(FEATURES)
                value, gradient = hinge(case, state['weights'], time_limit=contract['oracle_seconds'])
                delta = state['weights'] - anchor
                objective = float(value + .5 * (penalty * delta) @ delta)
                update(state, gradient + penalty * delta, contract['learning_rate'])
                progress['loss_sum'] += objective
                progress['position'] += 1
                if progress['position'] == len(records):
                    row = dict(epoch=progress['epoch']+1, steps=state['steps'], cases=len(records),
                        online_objective_mean=progress['loss_sum']/len(records),
                        displacement_norm=float(np.linalg.norm(state['weights']-anchor)))
                    progress['history'].append(row)
                    progress['epoch'] += 1
                    progress['position'], progress['loss_sum'] = 0, 0.
                    progress['order'] = rng.permutation(len(records)).tolist() if progress['epoch'] < contract['epochs'] else []
                    print(json.dumps(dict(event='epoch_complete', **row)), flush=True)
                    checkpoint()
                elif state['steps'] % 50 == 0:
                    checkpoint()
                    print(json.dumps(dict(event='checkpoint', steps=state['steps'], seconds=report['seconds'])), flush=True)
            if progress['epoch'] == contract['epochs']:
                weights = target / 'final-weights.npz'
                assert not weights.exists()
                np.savez_compressed(weights, weights=state['weights'], anchor=anchor, regularization=penalty,
                                    feature_names=np.asarray(FEATURES), max_fork_children=16)
                report.update(status='source_event_training_complete', model_fitted=True,
                    weights_sha256=sha(weights), quality_gain_established=False)
            checkpoint()
            print(json.dumps(report), flush=True)
        except BaseException as error:
            report.update(status='failed_requires_inspection', error=repr(error))
            checkpoint()
            raise


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
