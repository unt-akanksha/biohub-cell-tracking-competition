"""Stage verified exact-TTA caching on top of the existing offline overlap test."""
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_portable_adapter_v1 import replace_once
from research.submission_sharding import validate_submission_kernel_metadata


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior=ROOT/'.biohub/staging/biohub-trajectory-overlap-acceptance-v1'
    receipt=json.loads((ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-build.json').read_text())
    notebook_path=prior/'trajectory-overlap-acceptance.ipynb'
    assert sha(notebook_path)==receipt['notebook_sha256']
    assert sha(prior/'derived-contract.json')==receipt['contract_sha256']
    assert sha(prior/'overlays.json')==receipt['overlay_sha256']
    proof_path=ROOT/'reports/experiments/trajectory-overlap-cache-v1-result.json'
    proof=json.loads(proof_path.read_text());assert proof['status']=='exact_graph_identity_passed'
    assert len(proof['records'])==8 and len({r['movie'] for r in proof['records']})==4
    assert all(r['exact_graph_identity'] for r in proof['records'])
    source=ROOT/'.biohub/cache/trajectory-overlap-cache-v1-bundle'
    assert sha(source/'CONTRACT.json')==proof['contract_sha256']
    source_contract=json.loads((source/'CONTRACT.json').read_text())
    for name,digest in source_contract['bundle_sha256'].items():assert sha(source/name)==digest
    old_overlays=json.loads((prior/'overlays.json').read_text());overlays=dict(old_overlays)
    # Original disk bytes are checked above; inline notebook payloads use LF.
    # Require exact text identity after newline normalization, not a relaxed code diff.
    assert overlays['portable-worker.py']==(source/'portable-worker.py').read_text()
    overlays['public-predictor-original.py']=(source/'public-predictor-original.py').read_text(encoding='utf-8')
    assert overlays['public-predictor-original.py'].encode()==(source/'public-predictor-original.py').read_bytes().replace(b'\r\n',b'\n')
    overlays['NOTICE.txt']+='\nExact duplicate-view encode cache: preserves eight TTA contributions and their accumulation order. Four A10 complete movies and both emitted graphs per movie were byte-identical. This is not a D4 correction. Two-T4 acceptance remains required.\n'
    old_contract_text=(prior/'derived-contract.json').read_text()
    contract=json.loads(old_contract_text)
    for name,text in overlays.items():
        if name.endswith('.py'):ast.parse(text)
        contract['bundle_sha256'][name]=hashlib.sha256(text.encode()).hexdigest()
    contract.update(run_id='trajectory-kaggle-overlap-cache-v1',exact_duplicate_tta_encode_cache=True,
                    a10_cache_identity_sha256=sha(proof_path),production_runtime_acceptance_passed=False)
    contract_text=json.dumps(contract,indent=2);contract_sha=hashlib.sha256(contract_text.encode()).hexdigest()
    notebook=json.loads(notebook_path.read_text());code=''.join(notebook['cells'][1]['source'])
    code=replace_once(code,'_overlays = '+repr(old_overlays),'_overlays = '+repr(overlays))
    code=replace_once(code,'_derived_contract = '+repr(old_contract_text),'_derived_contract = '+repr(contract_text))
    code=replace_once(code,'CONTRACT_SHA256 = '+repr(receipt['contract_sha256']),'CONTRACT_SHA256 = '+repr(contract_sha))
    ast.parse(code)
    notebook['cells'][0]['source']=['Offline two-T4 overlap and exact duplicate-view caching acceptance; same weights, FP32 math and eight TTA contributions. No truth or submission generation.']
    notebook['cells'][1]['source']=code.splitlines(keepends=True)
    metadata=json.loads((prior/'kernel-metadata.json').read_text())
    metadata.update(id='indarkarhana/biohub-trajectory-overlap-cache-acceptance',title='Biohub Exact Cache Runtime Acceptance',code_file='trajectory-overlap-cache-acceptance.ipynb')
    validate_submission_kernel_metadata(metadata)
    target=ROOT/'.biohub/staging/biohub-trajectory-overlap-cache-acceptance-v1';target.mkdir(exist_ok=False)
    (target/metadata['code_file']).write_bytes(json.dumps(notebook,indent=2).encode())
    (target/'derived-contract.json').write_bytes(contract_text.encode())
    (target/'overlays.json').write_bytes(json.dumps(overlays,indent=2).encode())
    (target/'kernel-metadata.json').write_bytes(json.dumps(metadata,indent=2).encode())
    result=dict(status='staged_not_launched',kernel=metadata['id'],notebook_sha256=sha(target/metadata['code_file']),
                contract_sha256=sha(target/'derived-contract.json'),overlay_sha256=sha(target/'overlays.json'),
                source_proof_sha256=sha(proof_path),a10_verified_movies=4,a10_verified_graphs=8,
                a10_walltime_reduction_fraction=proof['walltime_reduction_fraction'],
                required_gpus=2,maximum_workers=4,acceptance_wall_cap_seconds=3600,
                fresh_quota_and_pending_reservation_check_required=True,sequential_gpu_gate_required=True,
                two_t4_acceptance_passed=False,kernel_pushed=False,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-overlap-cache-kaggle-v1-build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
