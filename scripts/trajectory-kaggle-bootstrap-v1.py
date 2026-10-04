"""Offline Kaggle entry point. Builder replaces only the four deployment constants."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

RUNTIME_ARCHIVE_SHA256 = '__ARCHIVE_SHA256__'
CONTRACT_SHA256 = '__CONTRACT_SHA256__'
RUN_MODE = '__RUN_MODE__'
RUNTIME_SLUG = 'biohub-trajectory-motion-runtime-v1'

started = time.monotonic()
if RUN_MODE not in ('acceptance', 'production'):
    raise ValueError('Unknown deployment mode')
if sys.version_info[:2] != (3, 12):
    raise ValueError('Frozen CP312 wheels require Python 3.12')
working = Path('/kaggle/working')
print(json.dumps(dict(event='bootstrap_started', mode=RUN_MODE)),flush=True)
input_root = Path('/kaggle/input')
runtime_candidates = [input_root/RUNTIME_SLUG,
                      input_root/'datasets/indarkarhana'/RUNTIME_SLUG]
# Kaggle expands uploaded ZIPs into dataset files (confirmed through files API).
# Do not recursively traverse hundreds of thousands of competition directories.
runtimes = [path for path in runtime_candidates if (path/'CONTRACT.json').is_file()]
if len(runtimes) != 1:
    raise ValueError('Exactly one attached expanded trajectory runtime required')
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()
runtime = runtimes[0]
if sha(runtime/'CONTRACT.json') != CONTRACT_SHA256:
    raise ValueError('Attached expanded dataset contract changed')
runtime_contract = json.loads((runtime/'CONTRACT.json').read_text())
if runtime_contract['ground_truth_included'] or runtime_contract['public_prediction_tables_included']:
    raise ValueError('Runtime must not contain labels or public prediction tables')
bundle = working / 'trajectory-runtime-v1'
bundle.mkdir(exist_ok=False)
for name, expected in runtime_contract['bundle_sha256'].items():
    source = runtime/name
    dest = bundle/name
    if (not source.resolve().is_relative_to(runtime.resolve())
            or not dest.resolve().is_relative_to(bundle.resolve()) or sha(source) != expected):
        raise ValueError('Changed or unsafe expanded runtime member')
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest)
shutil.copyfile(runtime/'CONTRACT.json',bundle/'CONTRACT.json')
print(json.dumps(dict(event='expanded_runtime_verified',files=len(runtime_contract['bundle_sha256']))),flush=True)
sys.path.insert(0, str(bundle))
from trajectory_runtime_v1 import verify_bundle
verify_bundle(bundle, CONTRACT_SHA256)
site = working / 'trajectory-site-v1'
site.mkdir(exist_ok=False)
wheels = sorted((bundle/'wheels').glob('*.whl'))
if len(wheels) != 64:
    raise ValueError('Exact offline wheel closure missing')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--no-compile', '--target', str(site), *map(str,wheels)], check=True, timeout=300)
env = dict(os.environ)
env.update(PYTHONPATH=os.pathsep.join((str(site),str(bundle))),
           OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',POLARS_MAX_THREADS='2',
           PYTHONUNBUFFERED='1',PIP_NO_INDEX='1')
competition_slug = 'biohub-cell-tracking-during-development'
competition_candidates = [input_root/competition_slug, input_root/'competitions'/competition_slug]
competition_roots = [p for p in competition_candidates
                     if (p/'sample_submission.csv').is_file() and (p/'test').is_dir() and (p/'train').is_dir()]
if len(competition_roots) != 1:
    raise ValueError('Exactly one Biohub competition input required')
data_root = competition_roots[0] / ('train' if RUN_MODE == 'acceptance' else 'test')
validation_movies = ['44b6_24264f12','44b6_81c256f0','6bba_23af9eeb','6bba_f1fde7e0',
                     '44b6_12dfb391','44b6_267148e4','6bba_062c8d37','6bba_07e24132']
all_movies = (validation_movies if RUN_MODE == 'acceptance'
              else sorted(p.stem for p in data_root.glob('*.zarr')))
if len(all_movies) < 2:
    raise ValueError('Two-GPU sharding requires at least two input movies')
smoke_movies = ['44b6_12dfb391','6bba_07e24132'] if RUN_MODE == 'acceptance' else all_movies[:2]
smoke_ids = working/'trajectory-smoke-movies.json'
smoke_ids.write_text(json.dumps(smoke_movies))

def run(mode, out, cap, movie_file=None, smoke_proof=None):
    command = [sys.executable,str(bundle/'run-trajectory-two-gpu-v1.py'),
               '--bundle',str(bundle),'--images',str(data_root),'--output',str(out),
               '--contract-sha256',CONTRACT_SHA256,'--mode',mode,'--wall-cap-seconds',str(cap)]
    if movie_file: command.extend(('--movies-json',str(movie_file)))
    if smoke_proof: command.extend(('--smoke-proof',str(smoke_proof)))
    # Controller's own watchdog reaps its workers before this outer deadline.
    subprocess.run(command, env=env, check=True, timeout=cap+60)

smoke_out = working/'trajectory-smoke'
run('smoke',smoke_out,600,movie_file=smoke_ids)
remaining = int((3300 if RUN_MODE=='acceptance' else 35900) - (time.monotonic()-started))
if remaining < 300:
    raise TimeoutError('Insufficient complete-run budget after smoke')
out = working/'trajectory-complete'
if RUN_MODE == 'acceptance':
    movie_file = working/'trajectory-validation-movies.json'
    movie_file.write_text(json.dumps(all_movies))
    run('validation',out,min(2600,remaining),movie_file=movie_file,smoke_proof=smoke_out/'result.json')
else:
    run('production',out,remaining,smoke_proof=smoke_out/'result.json')
    if (working/'submission.csv').exists():
        raise ValueError('Refusing to overwrite existing submission')
    shutil.copyfile(out/'submission.csv',working/'submission.csv')
receipt=json.loads((out/'result.json').read_text())
print(json.dumps(dict(status=receipt['status'],mode=RUN_MODE,
                     elapsed_seconds=time.monotonic()-started,
                     submission_created=RUN_MODE=='production',csv=receipt['csv']),indent=2))
