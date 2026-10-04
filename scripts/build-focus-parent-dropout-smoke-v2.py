"""Lossless packaging-only retry; identical worker and full audit bytes."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
OLD=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke.py'))
RUN='focus-parent-dropout-smoke-v2';SLUG='biohub-'+RUN
make_spec=OLD['make_spec']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_runtime(source):
    tree=ast.parse(source)
    values={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    runtime=ast.literal_eval(values['runtime_sources'])
    if 'packed_dropout_audit' in values:
        packed=ast.literal_eval(values['packed_dropout_audit'])
        runtime['dropout_audit.json']=gzip.decompress(base64.b64decode(packed,validate=True)).decode()
    return runtime


def build(spec=None):
    if spec is None:spec=make_spec()
    nb,meta=OLD['build'](spec);source=''.join(nb['cells'][1]['source'])
    original=decode_runtime(source);runtime=dict(original)
    audit=runtime.pop('dropout_audit.json')
    packed=base64.b64encode(gzip.compress(audit.encode(),mtime=0)).decode()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    replacement='import base64, gzip\npacked_dropout_audit = '+repr(packed)+'\nruntime_sources = '+repr(runtime)+"\nruntime_sources['dropout_audit.json'] = gzip.decompress(base64.b64decode(packed_dropout_audit)).decode()"
    source=source.replace(ast.get_source_segment(source,node),replacement)
    if decode_runtime(source)!=original:raise ValueError('Lossless full-runtime replay required')
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace(OLD['RUN'],RUN).replace('focus_parent_dropout_smoke','focus_parent_dropout_smoke_v2')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,scope='Lossless packaging-only retry of identical four-step dropout smoke')
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    if len(json.dumps(nb).encode())>=950000:raise ValueError('Notebook must remain below source size limit with margin')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite a frozen stage')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
