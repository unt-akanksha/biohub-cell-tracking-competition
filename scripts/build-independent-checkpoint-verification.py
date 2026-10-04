"""CPU-only Linux verification fallback for a host PyTorch DLL load stall."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-checkpoint-verification-v1'


def build():
    pilot = ROOT/'.biohub/cache/independent-real-pilot-v4-launch/biohub-independent-real-pilot-v1.ipynb'
    files = {'verify.py':(ROOT/'scripts/verify-independent-real-pilot.py').read_text(),
             'pilot_launch.ipynb':pilot.read_text()}
    source = '''import os, threading, json, hashlib, runpy
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
work = Path('/kaggle/working/checkpoint_verification')
work.mkdir(parents=True,exist_ok=True)
def timeout():
    (work/'timeout.json').write_text(json.dumps({'status':'timeout'}))
    os._exit(124)
watchdog = threading.Timer(600,timeout)
watchdog.daemon = True
watchdog.start()
'''
    source += 'files = '+repr(files)+'\n'
    source += '''for name,content in files.items():
    (work/name).write_text(content)
candidates = [Path('/kaggle/input')/p/'independent_real_pilot' for p in (
    'biohub-independent-real-pilot-v1',
    'notebooks/indarkarhana/biohub-independent-real-pilot-v1',
    'kernels/indarkarhana/biohub-independent-real-pilot-v1')]
root = next((p for p in candidates if (p/'outputs/last.pt').is_file()),None)
if root is None:
    raise RuntimeError('Version-four checkpoint input missing')
result = runpy.run_path(str(work/'verify.py'))['verify'](root,work/'pilot_launch.ipynb')
(work/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
watchdog.cancel()
'''
    ast.parse(source)
    nb = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(
        display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=source.splitlines(keepends=True))])
    metadata = dict(id='indarkarhana/'+TARGET.name,title='Biohub Independent Checkpoint Verification v1',
        code_file=TARGET.name+'.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=[],competition_sources=[],
        kernel_sources=['indarkarhana/biohub-independent-real-pilot-v1/4'])
    return nb,metadata


if __name__ == '__main__':
    notebook,metadata = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/metadata['code_file']).write_text(json.dumps(notebook))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(TARGET)
