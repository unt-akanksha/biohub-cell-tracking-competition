"""Actual full-trained first-fold model runtime plus approved fitting workload."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_pair_tree_inference import NativeResidual, predict

SMOKE = runpy.run_path(str(ROOT/'scripts/smoke-focus-pair-tree-inference.py'))
RUN = 'focus-tree-inference-workload-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workload(spec, original):
    roots = [ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
             ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    prior = {r['stem']: r for r in original['records']}
    rows = []
    for group, root in zip(spec['feature_groups'], roots):
        for row in group['feature_records']:
            if row['role'] != 'fitting':
                continue
            stem = row['stem']
            if stem not in prior:
                raise ValueError('Only the original12 fitting manifests may be opened')
            path = root/stem/'manifest.json'
            if sha(path) != prior[stem]['manifest_sha256'] or sha(path) != row['manifest_sha256']:
                raise ValueError('Exact verified fitting manifest required')
            manifest = json.loads(path.read_text())
            pairs = manifest['pairs']
            if manifest['role'] != 'fitting' or [p['source_frame'] for p in pairs] != list(range(99)):
                raise ValueError('All99 original fitting transitions required')
            if any(pairs[i]['target_nodes'] != pairs[i+1]['source_nodes'] for i in range(98)):
                raise ValueError('Original complete adjacent frame counts must agree')
            for p in pairs:
                if (p['target_nodes'] != p['known_parent']+p['known_absent']+p['unknown']
                        or max(p['source_nodes'], p['target_nodes']) > 2048):
                    raise ValueError('Complete target coverage and existing inference capacity required')
            all_choices = sum((p['source_nodes']+1)*p['target_nodes'] for p in pairs)
            known_choices = sum((p['source_nodes']+1)*(p['known_parent']+p['known_absent']) for p in pairs)
            if known_choices != prior[stem]['choices']:
                raise ValueError('Original known training coverage must replay')
            max_bound = max(min(32,p['target_nodes'])*(p['source_nodes']+1)*512*8
                            +(p['source_nodes']+p['target_nodes'])*128*8 for p in pairs)
            if max_bound > 1024**3:
                raise ValueError('Original complete fitting workload exceeds declared inference block budget')
            rows.append(dict(stem=stem, manifest_sha256=sha(path), transitions=99, complete_choices=all_choices,
                known_choices=known_choices, target_decisions=sum(p['target_nodes'] for p in pairs),
                unknown_targets=sum(p['unknown'] for p in pairs), maximum_conservative_block_bytes=max_bound,
                maximum_source_nodes=max(p['source_nodes'] for p in pairs), maximum_target_nodes=max(p['target_nodes'] for p in pairs)))
    if [r['stem'] for r in rows] != spec['contract']['fitting_stems'] or len(rows) != 12:
        raise ValueError('Exactly all12 original fitting movies required')
    return rows


def main():
    started = time.monotonic()
    report = ROOT/f'reports/experiments/{RUN}-profile.json'
    if report.exists():
        raise ValueError('Never overwrite completed profile')
    profile_path = ROOT/'reports/experiments/focus-pair-tree-first-fold-recovery-v1-profile.json'
    if sha(profile_path) != 'fefb07cf2f38aafa90eaab53d6f672a02d81b34e46869f4e326faa5c867400cb':
        raise ValueError('Completed full first-fold100tree fitting model required')
    profile = json.loads(profile_path.read_text())['worker']
    inference_path = ROOT/'reports/experiments/focus-pair-tree-inference-v1-smoke.json'
    if sha(inference_path) != '272b2e796c853871927306aa99eea44a30ffc281e0dd19b7c4129ca5828eebcf':
        raise ValueError('Original complete native inference smoke required')
    inference = json.loads(inference_path.read_text())
    if any(sha(ROOT/p) != v for p,v in {**profile['source_hashes'], **inference['source_hashes']}.items()):
        raise ValueError('Original checked model and inference implementations required')
    baseline_path = ROOT/'.biohub/cache/focus-pair-appearance-conditioned-lomo-v1/full/6bba_6479435d-model.json'
    tree_dir = ROOT/'.biohub/cache/focus-pair-tree-first-fold-recovery-v1/fit'
    if (sha(baseline_path) != profile['baseline_sha256'] or sha(tree_dir/'portable.json') != profile['tree_sha256']
            or sha(tree_dir/'native.ubj') != profile['native_sha256']):
        raise ValueError('Exact fixed first-fold model, not a quality-selected model required')
    saved = json.loads(baseline_path.read_text())
    if '6bba_57b7cc1e' not in saved['training_stems']:
        raise ValueError('Runtime packet must remain within this model fitting set')
    baseline, tree = saved['model'], json.loads((tree_dir/'portable.json').read_text())
    runtime = ROOT/'.biohub/cache/kernel-outputs/focus-pair-appearance-gpu-smoke-v1/focus_pair_appearance_gpu_smoke/runtime'
    if sha(runtime/'pair.npz') != inference['packet_sha256']:
        raise ValueError('Exact original real functionality packet required')
    with np.load(runtime/'pair.npz', allow_pickle=False) as data:
        packet = {k:data[k].copy() for k in data.files if k != 'labels'}
    parameters = json.loads((runtime/'spec.json').read_text())['motion_parameters']
    original = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in packet.items()}
    native = NativeResidual(tree, tree_dir/'native.ubj')
    tick = time.monotonic()
    out = predict(packet, parameters, baseline, native, 32)
    native_seconds = time.monotonic()-tick
    print(json.dumps(dict(stage='full_model_native_complete', native_seconds=native_seconds)), flush=True)
    tick = time.monotonic()
    reference = predict(packet, parameters, baseline, SMOKE['PortableResidual'](tree), 17)
    portable_seconds = time.monotonic()-tick
    for key in ('source_indices','target_indices','source_local_indices'):
        if not np.array_equal(out[key],reference[key]):
            raise ValueError('Every full-trained-model candidate decision must replay')
    errors = {k:float(np.max(np.abs(out[k]-reference[k]))) for k in ('selected_probability','null_probability')}
    if max(errors.values()) > 1e-8 or any(hashlib.sha256(v.tobytes()).hexdigest()!=original[k] for k,v in packet.items()):
        raise ValueError('Complete posterior replay or input preservation failed')
    spec_path = ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    original_path = ROOT/'reports/experiments/focus-candidate-ranker-v1-data-smoke.json'
    if (sha(spec_path) != '811dc9187cb587cbc803a0514c92cadb47e41704ef473c5532c269cb5790a36e') or sha(original_path) != 'c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf':
        raise ValueError('Original12 fitting-only feature inventory required')
    rows = workload(json.loads(spec_path.read_text()),json.loads(original_path.read_text()))
    total = sum(r['complete_choices'] for r in rows)
    choices = len(packet['target_coords'])*(len(packet['source_coords'])+1)
    result = dict(status='complete_full_model_runtime_and_fitting_workload_profile',
        source_sha256=sha(Path(__file__)), model_sha256=profile['tree_sha256'], baseline_sha256=profile['baseline_sha256'],
        native_seconds=native_seconds, portable_seconds=portable_seconds, measured_complete_choices=choices,
        maximum_probability_errors=errors, exact_all_target_decisions=True, rows=rows,
        total_fitting_choices=total, total_known_choices=sum(r['known_choices'] for r in rows),
        total_fitting_transitions=1188, maximum_conservative_block_bytes=max(r['maximum_conservative_block_bytes'] for r in rows),
        hypothetical_linear_work_scaled_seconds=total/choices*native_seconds,
        estimate_is_not_measured_movie_runtime=True, hidden_submission_runtime_established=False,
        diagnostic_movies_opened=0, source_movies_opened=0,new_target_movies_opened=0,
        gpu_seconds=0, quality_evaluated=False, authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    report.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,timeout=300,check=True)
    else:
        raise ValueError('No arguments supported')
