"""Stage exact-cache offline two-T4 validation with the frozen structured head."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_portable_adapter_v1 import replace_once
from research.submission_sharding import validate_submission_kernel_metadata
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior=ROOT/'.biohub/staging/biohub-trajectory-overlap-cache-acceptance-v1'
    previous=json.loads((ROOT/'reports/experiments/trajectory-overlap-cache-kaggle-v1-build.json').read_text())
    metadata=json.loads((prior/'kernel-metadata.json').read_text())
    assert sha(prior/metadata['code_file'])==previous['notebook_sha256']
    assert sha(prior/'derived-contract.json')==previous['contract_sha256']
    assert sha(prior/'overlays.json')==previous['overlay_sha256']
    portable=ROOT/'.biohub/cache/trajectory-structured-portable-v1'
    proof=json.loads((portable/'RESULT.json').read_text())
    assert proof['status']=='portable_inference_verified' and len(proof['records'])==8
    assert all(r['exact_graph_identity'] for r in proof['records'])
    assert sha(portable/'structured-trajectory.py')==proof['code_sha256']
    assert sha(portable/'structured-trajectory-model.json')==proof['model_sha256']
    old_overlays=json.loads((prior/'overlays.json').read_text());overlays=dict(old_overlays)
    overlays['structured-trajectory.py']=(portable/'structured-trajectory.py').read_text(encoding='utf-8')
    overlays['structured-trajectory-model.json']=(portable/'structured-trajectory-model.json').read_text(encoding='utf-8')
    worker=overlays['portable-worker.py']
    worker=replace_once(worker,"    motion = load_module(","    structured = load_module(args.bundle / 'structured-trajectory.py', 'structured_trajectory')\n    structured_model = json.loads((args.bundle / 'structured-trajectory-model.json').read_text())\n    if tuple(structured_model['features']) != structured.FEATURES:\n        raise ValueError('Structured feature schema changed')\n    motion = load_module(")
    anchor="                if helper.csv_equivalent_graph({int(k):v for k,v in repaired['nodes'].items()}, repaired['edges'], frames) != repaired:"
    replacement="""                (out / 'motion-repaired-prediction.json').write_text(json.dumps(repaired, sort_keys=True, allow_nan=False))
                raw_ilp = json.loads((out / 'pre-postprocess.json').read_text())
                repaired, structured_details = structured.refine(raw_ilp, repaired, coords, np.asarray(edges), structured_model['weights'])
                repair_details['structured_assignment'] = structured_details
"""+anchor
    worker=replace_once(worker,anchor,replacement);ast.parse(worker)
    overlays['portable-worker.py']=worker
    overlays['NOTICE.txt']+='\nOriginal source-trained 18-parameter structured trajectory assignment. Uses prediction-only physical/context features and preserves node coordinates/degrees/division/gap incident links. Source-fit regression is retained in research records; fixed model improved independent ten-movie confirmation and separate eight-movie validation without movie regressions. Runtime acceptance only, no submission in this notebook.\n'
    old_contract=(prior/'derived-contract.json').read_text();contract=json.loads(old_contract)
    for name,text in overlays.items():
        if name.endswith('.py'):ast.parse(text)
        contract['bundle_sha256'][name]=hashlib.sha256(text.encode()).hexdigest()
    contract.update(run_id='trajectory-structured-kaggle-v1',structured_trajectory_model=True,
                    structured_portable_proof_sha256=sha(portable/'RESULT.json'),
                    source_fit_movie_regression_recorded=True,production_runtime_acceptance_passed=False)
    contract_text=json.dumps(contract,indent=2);contract_sha=hashlib.sha256(contract_text.encode()).hexdigest()
    notebook=json.loads((prior/metadata['code_file']).read_text());code=''.join(notebook['cells'][1]['source'])
    code=replace_once(code,'_overlays = '+repr(old_overlays),'_overlays = '+repr(overlays))
    code=replace_once(code,'_derived_contract = '+repr(old_contract),'_derived_contract = '+repr(contract_text))
    code=replace_once(code,'CONTRACT_SHA256 = '+repr(previous['contract_sha256']),'CONTRACT_SHA256 = '+repr(contract_sha))
    ast.parse(code);notebook['cells'][1]['source']=code.splitlines(keepends=True)
    notebook['cells'][0]['source']=['Offline two-T4 end-to-end acceptance: unchanged public image models plus our fixed structured trajectory assignment. No annotations, no training, no submission generation.']
    metadata.update(id='indarkarhana/biohub-structured-trajectory-acceptance',title='Biohub Structured Trajectory Acceptance',code_file='structured-trajectory-acceptance.ipynb')
    validate_submission_kernel_metadata(metadata)
    target=ROOT/'.biohub/staging/biohub-structured-trajectory-acceptance-v1';target.mkdir(exist_ok=False)
    (target/metadata['code_file']).write_bytes(json.dumps(notebook,indent=2).encode())
    (target/'derived-contract.json').write_bytes(contract_text.encode())
    (target/'overlays.json').write_bytes(json.dumps(overlays,indent=2).encode())
    (target/'kernel-metadata.json').write_bytes(json.dumps(metadata,indent=2).encode())
    result=dict(status='staged_not_launched',kernel=metadata['id'],notebook_sha256=sha(target/metadata['code_file']),
                contract_sha256=sha(target/'derived-contract.json'),overlay_sha256=sha(target/'overlays.json'),
                required_gpus=2,maximum_workers=4,acceptance_wall_cap_seconds=3600,
                structured_portable_proof_sha256=sha(portable/'RESULT.json'),
                fresh_quota_and_pending_reservation_check_required=True,kernel_pushed=False,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-structured-kaggle-v1-build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
