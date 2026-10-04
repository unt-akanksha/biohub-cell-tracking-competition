"""Build a private offline one-hour pilot; no launch side effects."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'kaggle/biohub-independent-real-pilot-v1'


def build(steps=100):
    if steps not in (100,1000):
        raise ValueError('Only frozen 100-step smoke or 1000-step optimization profiles are allowed')
    vendor = ROOT / '.biohub/vendor/kaggle-cell-tracking-competition'
    files = {p.relative_to(vendor).as_posix(): p.read_text(encoding='utf-8')
             for p in (vendor / 'src').rglob('*.py')}
    for name in ('train_unet_transformer.py', 'predict_unet_transformer.py', 'evaluate.py', 'augmentations.py', 'dataspec.py'):
        files['scripts/'+name] = (vendor/'scripts'/name).read_text(encoding='utf-8')
    files['LICENSE'] = (vendor/'LICENSE').read_text(encoding='utf-8')
    local = {name: (ROOT/path).read_text(encoding='utf-8') for name, path in {
        'independent_real_baseline.py':'research/independent_real_baseline.py',
        'empty_graph_schema.py':'research/empty_graph_schema.py',
        'real_checkpoint_gpu_smoke.py':'research/real_checkpoint_gpu_smoke.py',
        'run_pilot.py':'scripts/run-independent-real-pilot.py',
        'split.json':'research/independent_real_baseline_v1_split.json'}.items()}
    previous = json.loads((ROOT/'kaggle/biohub-focus-raw-learned-linker-v1/biohub-focus-raw-learned-linker-v1.ipynb').read_text())
    dependencies = ''.join(previous['cells'][5]['source'])
    dependencies = dependencies[:dependencies.index('def remove_path(')]
    locator = next(n for n in ast.parse(dependencies).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'find_offline_package_dirs')
    dependencies = dependencies.replace(ast.get_source_segment(dependencies, locator),
        "def find_offline_package_dirs(artifacts):\n    return [artifacts / 'wheels']")
    # Only dependency functions are reused: no repo/model materialization.
    bootstrap = '''import os, sys, time, json, subprocess, hashlib, importlib, importlib.util
from pathlib import Path
started = time.monotonic()
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['POLARS_MAX_THREADS'] = '2'
os.environ['BIOHUB_ALLOW_PIP_INSTALL'] = '0'
work = Path('/kaggle/working/independent_real_pilot')
work.mkdir(parents=True, exist_ok=False)
repo = work/'repo'
runtime = work/'runtime'
import threading, signal
def hard_stop():
    child = globals().get('process')
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
    (work/'launcher_terminal.json').write_text(json.dumps(dict(
        status='hard_stop', run_id='independent-real-pilot-v1',
        elapsed_seconds=time.monotonic()-started, declared_budget_seconds=3600,
        submission_performed=False)))
    os._exit(124)
watchdog = threading.Timer(3540, hard_stop)
watchdog.daemon = True
watchdog.start()
'''
    write = f'''sources = {files!r}
runtime_sources = {local!r}
for folder, contents in ((repo, sources), (runtime, runtime_sources)):
    for name, content in contents.items():
        target = folder/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
manifest = {{name: hashlib.sha256(content.encode()).hexdigest() for name, content in sources.items()}}
(work/'source_hashes.json').write_text(json.dumps(manifest, indent=2))
(work/'runtime_hashes.json').write_text(json.dumps({{name: hashlib.sha256(content.encode()).hexdigest() for name, content in runtime_sources.items()}}, indent=2))
'''
    install = '''
support_candidates = [Path('/kaggle/input')/p for p in (
    'biohub-tracking-support-pack-50ep-v1',
    'datasets/pilkwang/biohub-tracking-support-pack-50ep-v1')]
support = next((p for p in support_candidates if (p/'wheels').is_dir()), None)
if support is None:
    raise RuntimeError('Exact offline support wheel mount not found')
ensure_dependencies(support)
'''
    launch = '''
data_candidates = [Path('/kaggle/input')/p/'train' for p in (
    'competitions/biohub-cell-tracking-during-development',
    'biohub-cell-tracking-during-development')]
data = next((p for p in data_candidates if p.is_dir()), None)
if data is None:
    raise RuntimeError('Competition training mount not found')
command = [sys.executable, '-u', str(runtime/'run_pilot.py'),
    '--repo', str(repo), '--runtime', str(runtime), '--manifest', str(runtime/'split.json'),
    '--data', str(data), '--output', str(work/'outputs')]
import signal
process = subprocess.Popen(command, start_new_session=True)
status = 'running'
try:
    code = process.wait(timeout=max(1, 3480-(time.monotonic()-started)))
    status = 'completed' if code == 0 else 'error'
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    status = 'hard_stop'
finally:
    watchdog.cancel()
    (work/'launcher_terminal.json').write_text(json.dumps(dict(status=status,
        run_id='independent-real-pilot-v1', elapsed_seconds=time.monotonic()-started,
        declared_budget_seconds=3600, submission_performed=False), indent=2))
if status != 'completed':
    raise RuntimeError('Pilot ended: '+status)
'''
    cells = []
    launch = launch.replace("'--data', str(data)", f"'--steps', '{steps}', '--data', str(data)")
    for source in (bootstrap, write, dependencies+install, launch):
        ast.parse(source)
        cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                          source=source.splitlines(keepends=True)))
    metadata = json.loads((ROOT/'kaggle/biohub-focus-raw-learned-linker-v1/kernel-metadata.json').read_text())
    metadata.update(id='indarkarhana/'+TARGET.name, title='Biohub Independent Real Pilot v1',
        code_file=TARGET.name+'.ipynb', kernel_sources=[],
        dataset_sources=['pilkwang/biohub-tracking-support-pack-50ep-v1'])
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        codex=dict(run_id='independent-real-pilot-v1', declared_budget_seconds=3600, max_steps=steps,
                   public_checkpoint_loaded=False, authorized_for_submission=False))), metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, choices=(100,1000), default=100)
    args = parser.parse_args()
    nb, metadata = build(args.steps)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET/metadata['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(TARGET)
