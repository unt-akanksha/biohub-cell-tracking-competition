"""Build revision2 using hard-linked immutable data, preserving failed revision1."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
old = ROOT/'.biohub/cache/visual-correspondence-ensemble-v1-bundle'
new = ROOT/'.biohub/cache/visual-correspondence-ensemble-v1-r2-bundle'
expected = '096abae5977aa95d68091cea49bafcdeed98bb879578b056e421b0472c291d4a'
if hashlib.sha256((old/'BUNDLE.json').read_bytes()).hexdigest() != expected:
    raise ValueError('Original immutable bundle changed')
new.mkdir(exist_ok=False)
contract = json.loads((old/'BUNDLE.json').read_text())
for record in contract['records']:
    source, dest = old/record['path'], new/record['path']
    if old.resolve() not in source.resolve().parents:
        raise ValueError('Invalid data path')
    if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Data changed')
    dest.parent.mkdir(exist_ok=True)
    os.link(source, dest)
for source, name in (
    ('research/visual_correspondence_data_v1.py', 'visual_correspondence_data_v1.py'),
    ('research/visual_correspondence_models_v1.py', 'visual_correspondence_models_v1.py'),
    ('scripts/run-visual-correspondence-ensemble-v1.py', 'run.py'),
):
    content = (ROOT/source).read_bytes(); (new/name).write_bytes(content)
    contract['files'][name] = hashlib.sha256(content).hexdigest()
contract.update(packaging_revision=2, previous_contract_sha256=expected,
                change='Guard None distance-baseline model before eval(); data/models/selection policy unchanged')
(new/'BUNDLE.json').write_text(json.dumps(contract, indent=2)+'\n')
print(json.dumps(dict(bundle=str(new), sha256=hashlib.sha256((new/'BUNDLE.json').read_bytes()).hexdigest())))
