"""One original fitting packet, complete label-free inference and bounded replay."""
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
from research.focus_candidate_ranker import candidate_arrays, pack
from research.focus_pair_appearance import descriptors, transform
from research.focus_pair_appearance_head import blocks
from research.focus_pair_appearance_inference import predict

RUN = 'focus-pair-appearance-inference-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-smoke.json'
    folder = ROOT / '.biohub/cache' / RUN
    if target.exists() or folder.exists():
        raise ValueError('Never overwrite completed or partial inference smoke')
    data_path = ROOT / 'reports/experiments/focus-pair-appearance-v1-data-smoke.json'
    if sha(data_path) != '4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f':
        raise ValueError('Pinned verified appearance data required')
    data = json.loads(data_path.read_text())
    runtime = ROOT / '.biohub/cache/kernel-outputs/focus-pair-appearance-gpu-smoke-v1/focus_pair_appearance_gpu_smoke/runtime'
    hashes = json.loads((runtime / 'source_hashes.json').read_text())
    if any(sha(runtime / name) != value for name, value in hashes.items()):
        raise ValueError('Verified original runtime and fitting packet required')
    for path in ('research/focus_candidate_ranker.py', 'research/focus_pair_appearance.py',
                 'research/focus_pair_appearance_head.py'):
        if sha(ROOT / path) != hashes[path]:
            raise ValueError('Original representation/model code changed')
    sources = {name: sha(ROOT / name) for name in (
        'research/focus_pair_appearance_inference.py', 'scripts/smoke-focus-pair-appearance-inference.py',
        'tests/test_focus_pair_appearance_inference.py', f'reports/experiments/{RUN}-design.md')}
    with np.load(runtime / 'pair.npz', allow_pickle=False) as saved:
        packet = {k: saved[k].copy() for k in saved.files}
    spec = json.loads((runtime / 'spec.json').read_text())
    parameters = spec['motion_parameters']
    ns, nt = len(packet['source_indices']), len(packet['target_indices'])
    input_array_hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in packet.items()}
    label_free = {k: v for k, v in packet.items() if k != 'labels'}
    known = np.flatnonzero(packet['labels'] >= 0)
    if len(known) != 20 or ns != 1160 or int(packet['source_frame']) != 31:
        raise ValueError('Exact preselected fitting-only functionality packet required')
    folder.mkdir()
    arms = []
    for row in data['stress']['arms']:
        arm = row['arm']
        model_path = ROOT / '.biohub/cache/focus-pair-appearance-v1' / (arm + '-smoke-model.json')
        if sha(model_path) != row['model_sha256']:
            raise ValueError('Frozen CPU smoke model required, not an unvalidated full fit')
        model = json.loads(model_path.read_text())
        theta = np.asarray(model['theta'])
        tick = time.monotonic()
        output = predict(label_free, parameters, model, target_block=32)
        seconds = time.monotonic()-tick
        maximum_error = 0.
        for first in range(0, nt, 17):
            columns = np.arange(first, min(first + 17, nt), dtype=np.int64)
            base = candidate_arrays(label_free, parameters, columns)
            extra = transform(descriptors(label_free, columns), base['null_rows'], model['projection'], arm)
            # Alternate split-matvec expression and different chunk boundaries.
            scores = (base['offset'] + base['features'] @ theta[:8] + extra @ theta[8:]).reshape(len(columns), ns+1)
            selected = scores.argmax(axis=1)
            maximum = scores.max(axis=1)
            normalizer = maximum + np.log(np.exp(scores-maximum[:, None]).sum(axis=1))
            probability = np.exp(scores[np.arange(len(columns)), selected]-normalizer)
            null = np.exp(-4.5-normalizer)
            if not np.array_equal(output['source_local_indices'][columns], np.where(selected == ns, -1, selected)):
                raise ValueError('Every target decision must match alternate bounded reference')
            np.testing.assert_allclose(output['selected_probability'][columns], probability, atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(output['null_probability'][columns], null, atol=1e-10, rtol=1e-9)
            maximum_error = max(maximum_error, float(np.max(np.abs(output['selected_probability'][columns]-probability))))
        packed = pack(packet, parameters, 'fitting')
        app = descriptors(packet, known)
        control = next(blocks([(packed, app)], model['projection'], arm))
        scores = (control['offset'] + control['x'] @ theta).reshape(len(known), ns+1)
        selected = scores.argmax(axis=1)
        if not np.array_equal(output['source_local_indices'][known], np.where(selected == ns, -1, selected)):
            raise ValueError('Every original known-target smoke decision must replay')
        selected_ids = np.full(nt, -1, np.int64)
        keep = output['source_local_indices'] >= 0
        selected_ids[keep] = packet['source_indices'][output['source_local_indices'][keep]]
        if not np.array_equal(selected_ids, output['source_indices']) or not np.array_equal(output['target_indices'], packet['target_indices']):
            raise ValueError('Every returned edge endpoint must preserve original node identity')
        output_path = folder / (arm + '-predictions.npz')
        np.savez_compressed(output_path, **output)
        with np.load(output_path, allow_pickle=False) as saved:
            if set(saved.files) != set(output) or any(not np.array_equal(saved[k], output[k]) for k in output):
                raise ValueError('Exact prediction reload differs')
        arms.append(dict(arm=arm, model_sha256=sha(model_path), output_sha256=sha(output_path),
                         source_nodes=ns, target_nodes=nt, complete_choices=nt*(ns+1),
                         originally_unknown_targets=nt-len(known), all_targets_replayed=True,
                         known_smoke_choices_replayed=True, maximum_probability_error=maximum_error,
                         inference_seconds=seconds, conservative_working_bytes=output['conservative_working_bytes']))
        print(json.dumps(arms[-1]), flush=True)
    if any(hashlib.sha256(v.tobytes()).hexdigest() != input_array_hashes[k] for k, v in packet.items()):
        raise ValueError('Input coordinates/features/identities/labels mutated')
    if any(sha(ROOT / p) != value for p, value in sources.items()):
        raise ValueError('Frozen inference method changed during smoke')
    result = dict(status='verified_complete_label_free_appearance_inference_smoke', run_id=RUN,
                  source_hashes=sources, packet_sha256=sha(runtime / 'pair.npz'), arms=arms,
                  input_arrays_unchanged=True, diagnostic_movies_opened=0, source_movies_opened=0,
                  new_target_movies_opened=0, gpu_seconds=0, quality_evaluated=False,
                  authorized_for_submission=False, elapsed_seconds=time.monotonic()-started)
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
