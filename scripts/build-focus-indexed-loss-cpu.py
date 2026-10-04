"""Standalone synthetic CPU test; no competition inputs, weights or GPU."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SLUG='biohub-focus-indexed-loss-cpu-v1'


def build():
    sources={name:(ROOT/path).read_text(encoding='utf-8') for name,path in {
        'focus_indexed_parent_loss.py':'research/focus_indexed_parent_loss.py',
        'sparse_parent_loss.py':'research/sparse_parent_loss.py',
        'sparse_parent_missing_null.py':'research/sparse_parent_missing_null.py',
        'probe.py':'scripts/probe-focus-indexed-loss.py'}.items()}
    wrapper='''import hashlib,json,subprocess,sys,time
from pathlib import Path
SOURCES = '''+repr(sources)+'''
runtime=Path('/kaggle/working/focus_indexed_loss_runtime')
runtime.mkdir(exist_ok=False)
for name,source in SOURCES.items():(runtime/name).write_text(source)
Path('/kaggle/working/focus_indexed_loss_source_hashes.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in SOURCES.items()},indent=2))
started=time.monotonic();status,error='failed',None
try:
    subprocess.run([sys.executable,str(runtime/'probe.py')],cwd='/kaggle/working',timeout=600,check=True)
    status='completed'
except Exception as exc:
    error=str(exc);raise
finally:
    Path('/kaggle/working/focus_indexed_loss_terminal.json').write_text(json.dumps(dict(status=status,error=error,elapsed_seconds=time.monotonic()-started,declared_budget_seconds=900,gpu_used=False)))
'''
    for source in [wrapper,*sources.values()]:ast.parse(source)
    metadata=dict(id='indarkarhana/'+SLUG,title=SLUG,code_file='run.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    return wrapper,metadata


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched CPU test')
    wrapper,metadata=build();target.mkdir()
    (target/'run.py').write_text(wrapper)
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(target)
