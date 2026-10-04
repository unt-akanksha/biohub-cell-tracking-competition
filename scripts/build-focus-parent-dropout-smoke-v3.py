"""Preserve original Windows audit bytes; retain all checksum guards."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
OLD=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke-v2.py'))
RUN='focus-parent-dropout-smoke-v3';SLUG='biohub-'+RUN
make_spec=OLD['make_spec'];decode_runtime=OLD['decode_runtime']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(spec=None):
    if spec is None:spec=make_spec()
    nb,meta=OLD['build'](spec);source=''.join(nb['cells'][1]['source'])
    runtime=decode_runtime(source)
    original=(ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json').read_bytes()
    if json.loads(original)!=json.loads(runtime['dropout_audit.json']):raise ValueError('Only newline representation may differ')
    expected_sha=hashlib.sha256(original).hexdigest()
    if spec.get('dropout_audit_sha256',expected_sha)!=expected_sha:raise ValueError('Original file-byte identity required')
    packed=base64.b64encode(gzip.compress(original,mtime=0)).decode()
    hashes=json.loads(runtime['source_hashes.json']);hashes['dropout_audit.json']=expected_sha
    runtime['source_hashes.json']=json.dumps(hashes);runtime.pop('dropout_audit.json')
    tree=ast.parse(source)
    replacements=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id=='packed_dropout_audit':replacements.append((ast.get_source_segment(source,node),'packed_dropout_audit = '+repr(packed)))
            if node.targets[0].id=='runtime_sources':replacements.append((ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)))
    for before,after in replacements:source=source.replace(before,after)
    if decode_runtime(source)['dropout_audit.json'].encode()!=original:raise ValueError('Exact original audit-byte round trip required')
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace(OLD['RUN'],RUN).replace('focus_parent_dropout_smoke_v2','focus_parent_dropout_smoke_v3')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,scope='Identical dropout smoke with exact original CRLF audit bytes')
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    if len(json.dumps(nb).encode())>=950000:raise ValueError('Source size bound required')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite frozen stage')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
