"""Stage a private self-contained300second GPU equivalence smoke; never launch here."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUN = 'focus-pair-appearance-gpu-smoke-v1'
SLUG = 'biohub-' + RUN


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    report = ROOT / 'reports/experiments'
    data_path = report / 'focus-pair-appearance-v1-data-smoke.json'
    verified_path = report / 'focus-pair-appearance-v1-verification.json'
    profile_path = report / 'focus-pair-appearance-v1-profile.json'
    for path, expected in [(data_path, '4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f'),
                           (verified_path, '823d8f64a4615ada45a220036a7125551ea3a997438d46b14d864bdaa988cab9'),
                           (profile_path, '28cee34f9d75a11c112668dbadf8d7626db96e80b4ad2e54783e0f42e57dd46c')]:
        if sha(path) != expected:
            raise ValueError('Actual complete CPU evidence required before GPU smoke')
    data, verified = json.loads(data_path.read_text()), json.loads(verified_path.read_text())
    if any(sha(ROOT / p) != value for p, value in data['source_hashes'].items()):
        raise ValueError('Frozen descriptor and CPU head methods changed')
    source_paths = ['research/focus_candidate_ranker.py', 'research/focus_pair_appearance.py',
                    'research/focus_pair_appearance_head.py', 'research/focus_pair_appearance_cuda.py',
                    'research/focus_pair_appearance_gpu_smoke.py']
    runtime = {name: (ROOT / name).read_bytes().decode('utf-8') for name in source_paths}
    runtime['research/__init__.py'] = ''
    for arm in data['stress']['arms']:
        name = arm['arm']
        path = ROOT / '.biohub/cache/focus-pair-appearance-v1' / (name + '-smoke-model.json')
        if sha(path) != arm['model_sha256']:
            raise ValueError('Actual CPU smoke coefficients required')
        runtime[name + '-cpu-model.json'] = path.read_bytes().decode('utf-8')
    prior = json.loads((report / 'focus-candidate-ranker-v1-data-smoke.json').read_text())
    record = next(r for r in prior['records'] if r['stem'] == '6bba_57b7cc1e')
    packet_path = ROOT / '.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs/6bba_57b7cc1e/031.npz'
    expected_packet = next(r['sha256'] for r in record['packet_hashes'] if r['file'] == '031.npz')
    if sha(packet_path) != expected_packet:
        raise ValueError('Exact original preselected fitting packet required')
    spec_path = ROOT / '.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    if sha(spec_path) != prior['training_spec_sha256']:
        raise ValueError('Original physical motion specification required')
    parameters = json.loads(spec_path.read_text())['motion_parameters']
    spec = dict(run_id=RUN, declared_budget_seconds=300, motion_parameters=parameters,
                stem='6bba_57b7cc1e', frame=31, packet_sha256=expected_packet,
                data_smoke_sha256=sha(data_path), verification_sha256=sha(verified_path),
                profile_sha256=sha(profile_path), design_sha256=sha(report / f'{RUN}-design.md'),
                controls={r['arm']: r['fitting_metrics'] for r in verified['smoke']})
    runtime['spec.json'] = json.dumps(spec, allow_nan=False)
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in runtime.items()}
    hashes['pair.npz'] = expected_packet
    runtime['source_hashes.json'] = json.dumps(hashes)
    payload = base64.b64encode(packet_path.read_bytes()).decode()
    init = '''import os, json, time, threading
from pathlib import Path
started = time.monotonic()
root = Path('/kaggle/working/focus_pair_appearance_gpu_smoke')
root.mkdir(exist_ok=False)
def emergency_stop():
    (root/'launcher_terminal.json').write_text(json.dumps(dict(status='watchdog', declared_budget_seconds=300, elapsed_seconds=time.monotonic()-started)))
    os._exit(124)
watchdog = threading.Timer(285, emergency_stop)
watchdog.daemon = True
watchdog.start()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
runtime = root/'runtime'
runtime.mkdir()
'''
    unpack = ('import base64, hashlib\nruntime_sources = ' + repr(runtime) + '\npacket_base64 = ' + repr(payload) + '''
for name, value in runtime_sources.items():
    path = runtime/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode('utf-8'))
(runtime/'pair.npz').write_bytes(base64.b64decode(packet_base64))
expected = json.loads((runtime/'source_hashes.json').read_text())
assert all(hashlib.sha256((runtime/name).read_bytes()).hexdigest()==value for name,value in expected.items())
print('Exact fitting packet and offline runtime ready', flush=True)
''')
    execute = '''import subprocess, sys
status = 'failed'
try:
    with (root/'worker.log').open('w') as log:
        process = subprocess.run([sys.executable,'-u','-m','research.focus_pair_appearance_gpu_smoke'], cwd=runtime,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=max(1,240-(time.monotonic()-started)), check=True)
    print((root/'worker.log').read_text(), flush=True)
    status = 'completed'
finally:
    watchdog.cancel()
    (root/'launcher_terminal.json').write_text(json.dumps(dict(status=status, declared_budget_seconds=300, elapsed_seconds=time.monotonic()-started, submission_performed=False)))
    if status != 'completed':
        print((root/'worker.log').read_text(), flush=True)
assert status == 'completed'
'''
    cells = []
    for code in (init, unpack, execute):
        ast.parse(code)
        cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=code.splitlines(keepends=True)))
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                            language_info=dict(name='python', version='3.11'),
                            codex=dict(run_id=RUN, declared_budget_seconds=300, scope='Fitting-packet CUDA equivalence only')))
    meta = dict(id='indarkarhana/' + SLUG, title=SLUG, code_file=SLUG + '.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
                dataset_sources=[], kernel_sources=[], model_sources=[],
                competition_sources=['biohub-cell-tracking-during-development'], machine_shape='NvidiaTeslaT4')
    if len(json.dumps(nb).encode()) >= 950000:
        raise ValueError('Bounded source payload required')
    return nb, meta


if __name__ == '__main__':
    target = ROOT / 'kaggle' / SLUG
    if target.exists():
        raise ValueError('Never overwrite frozen GPU stage')
    notebook, metadata = build()
    target.mkdir()
    (target / metadata['code_file']).write_text(json.dumps(notebook), encoding='utf-8')
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (target / 'staged_identity.json').write_text(json.dumps(dict(run_id=RUN, status='staged_not_launched',
        notebook_sha256=sha(target / metadata['code_file']), metadata_sha256=sha(target / 'kernel-metadata.json'),
        builder_sha256=sha(Path(__file__))), indent=2), encoding='utf-8')
    print(target)
