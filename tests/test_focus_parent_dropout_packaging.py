import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]


def test_lossless_packaging_all_runtime_bytes_and_caps():
    old=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke.py'))
    new=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke-v2.py'))
    nb1,_=old['build']({});nb2,meta=new['build']({})
    assert new['decode_runtime'](''.join(nb1['cells'][1]['source']))==new['decode_runtime'](''.join(nb2['cells'][1]['source']))
    assert len(json.dumps(nb2).encode())<950000
    assert nb2['metadata']['codex']['declared_budget_seconds']==900
    assert not meta['enable_internet'] and meta['enable_gpu'] and meta['is_private']
    for cell in nb2['cells']:ast.parse(''.join(cell['source']))


def test_original_audit_file_bytes_survive_windows_linux_transfer():
    import hashlib
    old=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke-v2.py'))
    new=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke-v3.py'))
    nb2,_=old['build']({});nb3,_=new['build']({})
    r2=old['decode_runtime'](''.join(nb2['cells'][1]['source']))
    r3=new['decode_runtime'](''.join(nb3['cells'][1]['source']))
    expected=(ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json').read_bytes()
    assert r3['dropout_audit.json'].encode()==expected
    assert json.loads(r2['dropout_audit.json'])==json.loads(r3['dropout_audit.json'])
    assert json.loads(r3['source_hashes.json'])['dropout_audit.json']==hashlib.sha256(expected).hexdigest()
    assert all(r2[k]==v for k,v in r3.items() if k not in ('dropout_audit.json','source_hashes.json'))
