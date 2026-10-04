"""Immutable standalone CPU check without external inputs or GPU use."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'biohub-blindspot-cpu-probe-v1'


def build():
    sources = {name:(ROOT/path).read_text() for name,path in {
        'blindspot_restoration.py':'research/blindspot_restoration.py',
        'probe-blindspot-restoration.py':'scripts/probe-blindspot-restoration.py'}.items()}
    wrapper = '''import hashlib,json,subprocess,sys,time
from pathlib import Path
SOURCES = ''' + repr(sources) + '''
root = Path('/kaggle/working/blindspot_cpu_runtime')
root.mkdir(exist_ok=False)
for name,source in SOURCES.items():
    (root/name).write_text(source)
Path('/kaggle/working/blindspot_source_hashes.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in SOURCES.items()},indent=2))
started = time.monotonic()
status,error = 'failed',None
try:
    subprocess.run([sys.executable,str(root/'probe-blindspot-restoration.py')],cwd='/kaggle/working',timeout=600,check=True)
    status = 'completed'
except Exception as exc:
    error = str(exc)
    raise
finally:
    Path('/kaggle/working/blindspot_cpu_terminal.json').write_text(json.dumps(dict(status=status,error=error,elapsed_seconds=time.monotonic()-started,declared_budget_seconds=900,gpu_hours=0,competition_data_read=False)))
'''
    for source in [wrapper,*sources.values()]:
        ast.parse(source)
    meta = dict(id='indarkarhana/'+SLUG,title=SLUG,code_file='run.py',language='python',
                kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,
                enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    return wrapper,meta


if __name__ == '__main__':
    target = ROOT/'kaggle'/SLUG
    if target.exists():
        raise ValueError('Do not overwrite an existing probe')
    wrapper,meta = build()
    target.mkdir()
    (target/'run.py').write_text(wrapper)
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
