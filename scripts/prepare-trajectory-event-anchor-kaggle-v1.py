"""Stage offline runtime acceptance only; preserve all candidate quality failures."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_portable_adapter_v1 import replace_once
from research.submission_sharding import validate_submission_kernel_metadata


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    prior=ROOT/'.biohub/staging/biohub-structured-trajectory-acceptance-v1'
    previous=read(ROOT/'reports/experiments/trajectory-structured-kaggle-v1-build.json')
    metadata=read(prior/'kernel-metadata.json')
    assert sha(prior/metadata['code_file'])==previous['notebook_sha256']
    assert sha(prior/'derived-contract.json')==previous['contract_sha256']
    assert sha(prior/'overlays.json')==previous['overlay_sha256']
    portable=ROOT/'.biohub/cache/trajectory-event-anchor-portable-v1';proof=read(portable/'RESULT.json')
    assert proof['status']=='portable_inference_verified' and len(proof['records'])==8
    assert all(r['exact_graph_identity'] and not r['budget_exhausted'] and not r['solver_fallbacks'] for r in proof['records'])
    assert proof['selection_failure_preserved'] and proof['validation_movie_failure_preserved']
    for filename,key in (('event-trajectory.py','code_sha256'),('event-trajectory-model.json','model_sha256')):
        assert sha(portable/filename)==proof[key]
    old_overlays=read(prior/'overlays.json');overlays=dict(old_overlays)
    for filename in ('event-trajectory.py','event-trajectory-model.json'):
        overlays[filename]=(portable/filename).read_text(encoding='utf-8')
    worker=overlays['portable-worker.py']
    worker=replace_once(worker,'    structured = load_module(',
        "    from threadpoolctl import threadpool_limits\n    event = load_module(args.bundle / 'event-trajectory.py', 'event_trajectory')\n    event_model = json.loads((args.bundle / 'event-trajectory-model.json').read_text())\n    if tuple(event_model['features']) != event.FEATURES:\n        raise ValueError('Event feature schema changed')\n    structured = load_module(")
    marker="                repair_details['structured_assignment'] = structured_details"
    worker=replace_once(worker,marker,marker+"\n"+"""                (out / 'structured-repaired-prediction.json').write_text(json.dumps(repaired, sort_keys=True, allow_nan=False))
                with threadpool_limits(limits=1, user_api='blas'):
                    repaired, event_details = event.refine(raw_ilp, repaired, coords, np.asarray(edges), event_model['weights'])
                repair_details['event_assignment'] = event_details""")
    ast.parse(worker);overlays['portable-worker.py']=worker
    overlays['NOTICE.txt']+='\nOriginal fixed event-assignment candidate, using the old source-trained edge weights and a source-only division prior. Prediction-only geometry and neural evidence; no movie-ID router or training annotations. Source, selection, and validation results are recorded separately. Selection embryo/movie and validation movie regressions remain explicitly recorded, not waived by this runtime test. This notebook performs acceptance only, not competition submission.\n'
    old_contract=(prior/'derived-contract.json').read_text(encoding='utf-8');contract=json.loads(old_contract)
    for name,text in overlays.items():
        if name.endswith('.py'):ast.parse(text)
        contract['bundle_sha256'][name]=hashlib.sha256(text.encode()).hexdigest()
    contract.update(run_id='trajectory-event-anchor-kaggle-v1',event_anchor_model=True,
        event_portable_proof_sha256=sha(portable/'RESULT.json'),
        event_selection_failure_preserved=True,event_validation_movie_failure_preserved=True,
        event_runtime_acceptance_passed=False,event_authorized_for_submission=False)
    contract_text=json.dumps(contract,indent=2);contract_sha=hashlib.sha256(contract_text.encode()).hexdigest()
    notebook=read(prior/metadata['code_file']);code=''.join(notebook['cells'][1]['source'])
    code=replace_once(code,'_overlays = '+repr(old_overlays),'_overlays = '+repr(overlays))
    code=replace_once(code,'_derived_contract = '+repr(old_contract),'_derived_contract = '+repr(contract_text))
    code=replace_once(code,'CONTRACT_SHA256 = '+repr(previous['contract_sha256']),'CONTRACT_SHA256 = '+repr(contract_sha))
    ast.parse(code);notebook['cells'][1]['source']=code.splitlines(keepends=True)
    notebook['cells'][0]['source']=['Offline two-T4 acceptance of our fixed event-assignment candidate. No labels, training, model selection, or competition submission. Recorded quality regressions are not waived.']
    metadata.update(id='indarkarhana/biohub-event-anchor-acceptance',title='Biohub Event Anchor Acceptance',code_file='event-anchor-acceptance.ipynb')
    validate_submission_kernel_metadata(metadata)
    target=ROOT/'.biohub/staging/biohub-event-anchor-acceptance-v1';target.mkdir(exist_ok=False)
    (target/metadata['code_file']).write_bytes(json.dumps(notebook,indent=2).encode())
    (target/'derived-contract.json').write_bytes(contract_text.encode())
    (target/'overlays.json').write_bytes(json.dumps(overlays,indent=2).encode())
    (target/'kernel-metadata.json').write_bytes(json.dumps(metadata,indent=2).encode())
    result=dict(status='staged_not_launched',kernel=metadata['id'],notebook_sha256=sha(target/metadata['code_file']),
        contract_sha256=sha(target/'derived-contract.json'),overlay_sha256=sha(target/'overlays.json'),
        required_gpus=2,maximum_workers=4,acceptance_wall_cap_seconds=3600,
        event_portable_proof_sha256=sha(portable/'RESULT.json'),quality_failures_preserved=True,
        fresh_quota_and_pending_reservation_check_required=True,kernel_pushed=False,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-build.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
