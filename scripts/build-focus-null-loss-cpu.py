"""CPU-only Torch supervision smoke; no inputs, GPU, training data or internet."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SLUG='biohub-focus-null-loss-cpu-v1'


def build():
    paths={'probe.py':'scripts/probe-focus-null-balanced-loss.py'}
    for name in ('focus_null_balanced_loss','focus_indexed_parent_loss','sparse_parent_loss','sparse_parent_missing_null'):
        paths[name+'.py']='research/'+name+'.py'
    sources={name:(ROOT/path).read_text(encoding='utf-8') for name,path in paths.items()}
    wrapper='''import hashlib,json,subprocess,sys,time
from pathlib import Path
SOURCES = '''+repr(sources)+'''
runtime=Path('/kaggle/working/focus_null_loss_runtime');runtime.mkdir(exist_ok=False)
for name,source in SOURCES.items():(runtime/name).write_text(source)
Path('/kaggle/working/focus_null_loss_source_hashes.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in SOURCES.items()},indent=2))
started=time.monotonic();status,error='failed',None
try:
    subprocess.run([sys.executable,str(runtime/'probe.py')],cwd=runtime,timeout=600,check=True)
    status='completed'
except Exception as exc:
    error=str(exc);raise
finally:
    Path('/kaggle/working/focus_null_loss_terminal.json').write_text(json.dumps(dict(status=status,error=error,elapsed_seconds=time.monotonic()-started,declared_budget_seconds=900,gpu_used=False)))
'''
    for source in [wrapper,*sources.values()]:ast.parse(source)
    metadata=dict(id='indarkarhana/'+SLUG,title=SLUG,code_file='run.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    return wrapper,metadata


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched CPU smoke')
    source,meta=build();target.mkdir();(target/'run.py').write_text(source)
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(path=str(target),source_sha256=hashlib.sha256((target/'run.py').read_bytes()).hexdigest())))
