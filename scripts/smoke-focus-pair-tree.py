"""Fixed real fitting packet: nonlinear ranking functionality, not promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import pack
from research.focus_pair_appearance import descriptors
from research.focus_pair_appearance_head import blocks
from research.focus_pair_tree import fit, portable_predict, derivatives, SETTINGS, VERSION, ROUNDS

RUN = 'focus-pair-tree-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-smoke.json'
    cache = ROOT / '.biohub/cache' / RUN
    if target.exists() or cache.exists():
        raise ValueError('Never overwrite partial or completed real smoke')
    runtime = ROOT / '.biohub/cache/kernel-outputs/focus-pair-appearance-gpu-smoke-v1/focus_pair_appearance_gpu_smoke/runtime'
    hashes = json.loads((runtime / 'source_hashes.json').read_text())
    if any(sha(runtime / p) != value for p, value in hashes.items()):
        raise ValueError('Original packet/runtime hash identity required')
    for name in ('research/focus_candidate_ranker.py', 'research/focus_pair_appearance.py',
                 'research/focus_pair_appearance_head.py'):
        if sha(ROOT / name) != hashes[name]:
            raise ValueError('Original pair representation code changed')
    names = ['research/focus_pair_tree.py', 'scripts/smoke-focus-pair-tree.py',
             'tests/test_focus_pair_tree.py', f'reports/experiments/{RUN}-design.md']
    frozen = {p: sha(ROOT / p) for p in names}
    with np.load(runtime / 'pair.npz', allow_pickle=False) as data:
        packet = {k: data[k].copy() for k in data.files}
    original = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in packet.items()}
    parameters = json.loads((runtime / 'spec.json').read_text())['motion_parameters']
    model_path = runtime / 'full-cpu-model.json'
    if sha(model_path) != '760c4af100278bc5431d20039f20d73963f078228cd3310dec7843fb29e5a32e':
        raise ValueError('Original full CPU smoke baseline required')
    base_model = json.loads(model_path.read_text())
    known = np.flatnonzero(packet['labels'] >= 0)
    base = pack(packet, parameters, 'fitting')
    appearance = descriptors(packet, known)
    prepared = list(blocks([(base, appearance)], base_model['projection'], 'full'))
    if len(prepared) != 1 or len(known) != 20 or len(base['offset']) != 23220:
        raise ValueError('Only the original20 complete fitting groups required')
    block = prepared[0]
    margin = block['offset'] + block['x'] @ np.asarray(base_model['theta'])
    group_args = [block[k] for k in ('starts', 'sizes', 'chosen', 'present')]
    initial, grad, _ = derivatives(margin, *group_args, base_model['absent_weight'])
    direction = np.sin(np.arange(len(margin))+.5)
    direction /= np.linalg.norm(direction)
    eps = 1e-4
    fd = (derivatives(margin+eps*direction, *group_args, base_model['absent_weight'])[0]
          - derivatives(margin-eps*direction, *group_args, base_model['absent_weight'])[0])/(2*eps)
    if abs(fd-grad @ direction) > 1e-7:
        raise ValueError('Actual real-packet gradient finite difference failed')
    cache.mkdir()
    tick = time.monotonic()
    tree = fit(block['x'], margin, *group_args, base_model['absent_weight'], cache/'fit', 'fitting',
               progress=lambda r: print(json.dumps(r), flush=True))
    seconds = time.monotonic()-tick
    residual = portable_predict(tree, block['x'])
    score = margin+residual
    null_values = residual[block['null_rows']]
    if not np.all(null_values == null_values[0]):
        raise ValueError('Identical null feature rows must have identical residuals')
    # Label-free exported prediction requires only feature matrix plus external baseline.
    reloaded = json.loads((cache/'fit/portable.json').read_text())
    replay = margin+portable_predict(reloaded, np.array(block['x'], copy=True))
    if not np.array_equal(score, replay):
        raise ValueError('Actual portable score replay failed')
    shifted = score-null_values[0]
    np.testing.assert_allclose(shifted[block['null_rows']], -4.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(derivatives(score, *group_args, base_model['absent_weight'])[0],
                               derivatives(shifted, *group_args, base_model['absent_weight'])[0], atol=1e-10)
    before = margin.reshape(20, -1).argmax(axis=1)
    after = score.reshape(20, -1).argmax(axis=1)
    truth = block['chosen']-block['starts']
    if any(hashlib.sha256(v.tobytes()).hexdigest() != original[k] for k, v in packet.items()):
        raise ValueError('Source packet was mutated')
    if any(sha(ROOT / p) != value for p, value in frozen.items()):
        raise ValueError('Frozen smoke source changed')
    np.savez_compressed(cache/'smoke-scores.npz', margin=margin, residual=residual, score=score,
                        starts=block['starts'], sizes=block['sizes'], chosen=block['chosen'])
    result = dict(status='completed_real_pair_tree_functionality_smoke', source_hashes=frozen,
                  packet_sha256=sha(runtime/'pair.npz'), baseline_sha256=sha(model_path),
                  tree_sha256=sha(cache/'fit/portable.json'), native_sha256=sha(cache/'fit/native.ubj'),
                  scores_sha256=sha(cache/'smoke-scores.npz'), settings=SETTINGS, version=VERSION,
                  rounds=ROUNDS, groups=20, choices=len(score), gradient_fd_error=abs(fd-grad @ direction),
                  initial_fitting_loss=initial, final_fitting_loss=tree['final_loss'],
                  fitting_correct_before=int((before == truth).sum()), fitting_correct_after=int((after == truth).sum()),
                  native_portable_max_error=tree['native_portable_max_error'], native_checkpoints=4,
                  fit_seconds=seconds, input_packet_unchanged=True, label_free_score_replay=True,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  gpu_seconds=0, quality_evaluated=False, authorized_for_submission=False,
                  elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=300, check=True)
    else:
        raise ValueError('No arguments supported')
