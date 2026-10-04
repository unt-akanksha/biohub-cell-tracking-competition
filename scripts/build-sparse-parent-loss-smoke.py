"""CPU-only numerical/gradient gate, no data or checkpoint access."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-sparse-parent-loss-smoke-v1'


def build():
    files = {p:(ROOT/p).read_text() for p in (
        'research/sparse_parent_loss.py','tests/test_sparse_parent_loss.py')}
    code = '''import os, json, subprocess, sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
work = Path('/kaggle/working/parent_loss_smoke')
work.mkdir(parents=True,exist_ok=True)
'''
    code += 'files = '+repr(files)+'\n'
    code += '''for name,content in files.items():
    path = work/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content)
result = subprocess.run([sys.executable,'-m','pytest',str(work/'tests/test_sparse_parent_loss.py'),'-q'],
    cwd=work,timeout=180,text=True,capture_output=True)
print(result.stdout,flush=True)
print(result.stderr,flush=True)
(work/'test_result.json').write_text(json.dumps(dict(exit_code=result.returncode,
    stdout=result.stdout,stderr=result.stderr,cpu_only=True,authorized_for_submission=False),indent=2))
assert result.returncode == 0
'''
    ast.parse(code)
    nb = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(
        display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    meta = dict(id='indarkarhana/'+TARGET.name,title='Biohub Sparse Parent Loss Smoke v1',
        code_file=TARGET.name+'.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        kernel_sources=[],dataset_sources=[],competition_sources=[])
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
