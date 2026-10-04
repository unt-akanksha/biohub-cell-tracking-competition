"""Whole original frame pair, all targets: native throughput and portable replay."""
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
from research.focus_pair_tree_inference import NativeResidual, predict
from research.focus_pair_tree import portable_predict

RUN = 'focus-pair-tree-inference-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PortableResidual:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        return portable_predict(self.model, features)


def main():
    started = time.monotonic()
    output = ROOT/f'reports/experiments/{RUN}-smoke.json'
    cache = ROOT/'.biohub/cache'/RUN
    if output.exists() or cache.exists():
        raise ValueError('Never overwrite completed or partial inference smoke')
    smoke_path = ROOT/'reports/experiments/focus-pair-tree-v1-smoke.json'
    if sha(smoke_path) != '41fd6c6e9d8ea8cfa67a0b6d6d4917958e430fd543eaaebf1fac3e3e70836e2c':
        raise ValueError('Original verified20-group tree smoke required')
    smoke = json.loads(smoke_path.read_text())
    if any(sha(ROOT/p) != v for p, v in smoke['source_hashes'].items()):
        raise ValueError('Original frozen tree implementation changed')
    runtime = ROOT/'.biohub/cache/kernel-outputs/focus-pair-appearance-gpu-smoke-v1/focus_pair_appearance_gpu_smoke/runtime'
    runtime_hashes = json.loads((runtime/'source_hashes.json').read_text())
    if any(sha(runtime/p) != value for p, value in runtime_hashes.items()):
        raise ValueError('Original real packet and CPU baseline identity required')
    if sha(runtime/'full-cpu-model.json') != smoke['baseline_sha256'] or sha(runtime/'pair.npz') != smoke['packet_sha256']:
        raise ValueError('Exact original smoke data required')
    tree_dir = ROOT/'.biohub/cache/focus-pair-tree-v1/fit'
    if sha(tree_dir/'portable.json') != smoke['tree_sha256'] or sha(tree_dir/'native.ubj') != smoke['native_sha256']:
        raise ValueError('Exact original native/portable smoke trees required')
    baseline = json.loads((runtime/'full-cpu-model.json').read_text())
    tree = json.loads((tree_dir/'portable.json').read_text())
    parameters = json.loads((runtime/'spec.json').read_text())['motion_parameters']
    names = ['research/focus_pair_tree_inference.py', 'scripts/smoke-focus-pair-tree-inference.py',
             'tests/test_focus_pair_tree_inference.py', f'reports/experiments/{RUN}-design.md',
             'research/focus_candidate_ranker.py', 'research/focus_pair_appearance.py',
             'research/focus_pair_appearance_inference.py', 'research/focus_pair_tree.py']
    frozen = {p: sha(ROOT/p) for p in names}
    with np.load(runtime/'pair.npz', allow_pickle=False) as data:
        packet = {k: data[k].copy() for k in data.files}
    hashes = {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in packet.items()}
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    if ns != 1160 or nt != 1187 or int(packet['source_frame']) != 31:
        raise ValueError('Only original preselected fitting functionality frame allowed')
    known = np.flatnonzero(packet['labels'] >= 0)
    label_free = {k: v for k, v in packet.items() if k != 'labels'}
    residual = NativeResidual(tree, tree_dir/'native.ubj')
    cache.mkdir()
    tick = time.monotonic()
    native = predict(label_free, parameters, baseline, residual, target_block=32)
    native_seconds = time.monotonic()-tick
    print(json.dumps(dict(stage='native_complete', native_seconds=native_seconds, targets=nt)), flush=True)
    tick = time.monotonic()
    portable = predict(label_free, parameters, baseline, PortableResidual(tree), target_block=17)
    portable_seconds = time.monotonic()-tick
    for key in ('source_indices', 'target_indices', 'source_local_indices'):
        if not np.array_equal(native[key], portable[key]):
            raise ValueError('Every native/portable global node decision must match')
    error = 0.
    for key in ('selected_probability', 'null_probability'):
        np.testing.assert_allclose(native[key], portable[key], rtol=1e-7, atol=1e-8)
        error = max(error, float(np.max(np.abs(native[key]-portable[key]))))
    scores_path = ROOT/'.biohub/cache/focus-pair-tree-v1/smoke-scores.npz'
    if sha(scores_path) != smoke['scores_sha256']:
        raise ValueError('Original known-group scores required')
    with np.load(scores_path, allow_pickle=False) as saved:
        known_scores = saved['score'].reshape(20, ns+1)
    choices = known_scores.argmax(axis=1)
    if not np.array_equal(native['source_local_indices'][known], np.where(choices == ns, -1, choices)):
        raise ValueError('Every original known-group smoke decision must replay')
    if any(hashlib.sha256(a.tobytes()).hexdigest() != hashes[k] for k, a in packet.items()):
        raise ValueError('Input packet mutated')
    prediction_path = cache/'native-predictions.npz'
    np.savez_compressed(prediction_path, **native)
    with np.load(prediction_path, allow_pickle=False) as saved:
        if set(saved.files) != set(native) or any(not np.array_equal(saved[k], native[k]) for k in saved.files):
            raise ValueError('Exact serialized inference output required')
    if any(sha(ROOT/p) != v for p, v in frozen.items()):
        raise ValueError('Frozen inference method changed')
    result = dict(status='verified_complete_native_tree_inference_smoke', source_hashes=frozen,
        packet_sha256=sha(runtime/'pair.npz'), baseline_sha256=smoke['baseline_sha256'],
        tree_sha256=smoke['tree_sha256'], native_model_sha256=smoke['native_sha256'],
        predictions_sha256=sha(prediction_path), source_nodes=ns, target_nodes=nt,
        complete_choices=nt*(ns+1), originally_unknown_targets=nt-len(known),
        every_target_native_portable_replayed=True, known_smoke_choices_replayed=True,
        native_seconds=native_seconds, portable_seconds=portable_seconds,
        maximum_probability_error=error, conservative_working_bytes=native['conservative_working_bytes'],
        memory_bound_is_estimate_not_measured_rss=True, original_nodes_and_input_unchanged=True,
        diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
        gpu_seconds=0, quality_evaluated=False, authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=300, check=True)
    else:
        raise ValueError('No arguments supported')
