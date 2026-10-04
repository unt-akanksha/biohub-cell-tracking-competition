"""Static byte-level overlay validation; no notebook cells or Kaggle calls."""
import ast
import hashlib
import json
from pathlib import Path
from research.submission_sharding import validate_submission_kernel_metadata

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'.biohub/staging/biohub-trajectory-overlap-cache-acceptance-v1'


def digest(data):return hashlib.sha256(data).hexdigest()


def test_inline_payload_is_exact_contract_and_overlay_bytes():
    notebook=json.loads((STAGE/'trajectory-overlap-cache-acceptance.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(''.join(notebook['cells'][1]['source']))
    values={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('_overlays','_derived_contract'):
                assert name not in values;values[name]=ast.literal_eval(node.value)
    assert values['_derived_contract'].encode()==(STAGE/'derived-contract.json').read_bytes()
    assert values['_overlays']==json.loads((STAGE/'overlays.json').read_text(encoding='utf-8'))
    contract=json.loads(values['_derived_contract'])
    for name,text in values['_overlays'].items():
        assert Path(name).name==name
        assert digest(text.encode())==contract['bundle_sha256'][name]
        if name.endswith('.py'):ast.parse(text)


def test_cached_predictor_matches_a10_verified_source_and_assets_unchanged():
    overlays=json.loads((STAGE/'overlays.json').read_text(encoding='utf-8'))
    source=ROOT/'.biohub/cache/trajectory-overlap-cache-v1-bundle'
    assert overlays['public-predictor-original.py'].encode()==(source/'public-predictor-original.py').read_bytes().replace(b'\r\n',b'\n')
    contract=json.loads((STAGE/'derived-contract.json').read_text())
    base=json.loads((ROOT/'.biohub/cache/trajectory-kaggle-runtime-v2-bundle/CONTRACT.json').read_text())
    for name,value in base['bundle_sha256'].items():
        if name not in overlays:assert contract['bundle_sha256'][name]==value
    assert not contract['ground_truth_included'] and not contract['public_prediction_tables_included']
    assert not contract['production_runtime_acceptance_passed']


def test_private_offline_metadata_and_unlaunched_receipt():
    metadata=json.loads((STAGE/'kernel-metadata.json').read_text())
    validate_submission_kernel_metadata(metadata)
    assert metadata['is_private'] and not metadata['enable_internet'] and not metadata['enable_tpu']
    receipt=json.loads((ROOT/'reports/experiments/trajectory-overlap-cache-kaggle-v1-build.json').read_text())
    assert digest((STAGE/'derived-contract.json').read_bytes())==receipt['contract_sha256']
    assert digest((STAGE/metadata['code_file']).read_bytes())==receipt['notebook_sha256']
    assert not receipt['kernel_pushed'] and not receipt['submission_performed']
