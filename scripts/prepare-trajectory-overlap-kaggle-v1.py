"""Embed small verified scheduling overlays; reuse the accepted offline dataset."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,verify_bundle
from research.trajectory_portable_adapter_v1 import replace_once
from research.submission_sharding import validate_submission_kernel_metadata


def main():
    proof_path=ROOT/'reports/experiments/trajectory-overlap-v1-result.json'
    proof=json.loads(proof_path.read_text())
    if (proof['status']!='exact_graph_identity_passed' or len(proof['records'])!=8
            or not all(r['exact_graph_identity'] for r in proof['records'])
            or proof['contract_sha256']!='cb3a8e74a63f69ac270ea5c6995c7260dcd80ea15dcbd1faefa95586681cef1c'):
        raise ValueError('Actual A10 overlap identity gate required')
    base=ROOT/'.biohub/cache/trajectory-kaggle-runtime-v2-bundle'
    base_contract='d10d19b1e21b30e2eb5c6bf1c05161559f74594ce30e250e3c473f2d8b0379f2'
    contract=verify_bundle(base,base_contract)
    runner=(base/'run-trajectory-two-gpu-v1.py').read_text()
    runner=replace_once(runner,'\n\ndef main():',
        '\n\nfrom trajectory_work_slots_v1 import (build_movie_slots as build_movie_shards,\n'
        '    validate_slot_outputs as validate_shard_outputs)\n\ndef main():')
    runner=replace_once(runner,"                   shard_plan_sha256=shard_plan_sha256(shards),",
                         "                   worker_count=len(shards),maximum_workers_per_gpu=2,\n"
                         "                   shard_plan_sha256=shard_plan_sha256(shards),")
    runner=replace_once(runner,"env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',\n                       POLARS_MAX_THREADS='2', PYTHONUNBUFFERED='1')",
                         "env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',\n                       POLARS_MAX_THREADS='1', PYTHONUNBUFFERED='1')")
    worker=(ROOT/'.biohub/cache/trajectory-overlap-v1-bundle/portable-worker.py').read_text()
    overlays={'portable-worker.py':worker,'run-trajectory-two-gpu-v1.py':runner,
              'trajectory_work_slots_v1.py':(ROOT/'research/trajectory_work_slots_v1.py').read_text(),
              'NOTICE.txt':(base/'NOTICE.txt').read_text()+'\nScheduling revision: at most two FP32 whole-movie workers per physical GPU; one CPU thread and35% allocator each. No model or precision change. A10 four-movie outputs are identical to the accepted FP32 graphs. Two-T4 acceptance still required.\n'}
    for name,text in overlays.items():
        if name.endswith('.py'):ast.parse(text)
        contract['bundle_sha256'][name]=hashlib.sha256(text.encode('utf-8')).hexdigest()
    contract.update(run_id='trajectory-kaggle-overlap-v1',base_contract_sha256=base_contract,
                    scheduling_revision=1,maximum_workers_per_gpu=2,
                    a10_overlap_identity_sha256=sha(proof_path),production_runtime_acceptance_passed=False)
    contract_text=json.dumps(contract,indent=2)
    contract_sha=hashlib.sha256(contract_text.encode('utf-8')).hexdigest()
    out=ROOT/'.biohub/staging/biohub-trajectory-overlap-acceptance-v1'
    out.mkdir(exist_ok=False)
    (out/'derived-contract.json').write_bytes(contract_text.encode('utf-8'))
    (out/'overlays.json').write_bytes(json.dumps(overlays,indent=2).encode('utf-8'))
    base_notebook=ROOT/'.biohub/staging/biohub-trajectory-motion-acceptance-v3/trajectory-motion-acceptance.ipynb'
    if sha(base_notebook)!='410061255f3888a031283be604c7b60bf1d5971bd48c5866327e7baf038bf3de':
        raise ValueError('Accepted bootstrap changed')
    notebook=json.loads(base_notebook.read_text())
    code=''.join(notebook['cells'][1]['source'])
    insertion='''
# Verified scheduling overlay; base dataset, weights and wheel closure unchanged.
_overlays = OVERLAY_PAYLOAD
_derived_contract = DERIVED_CONTRACT_PAYLOAD
for _name,_text in _overlays.items():
    _dest = bundle/_name
    if Path(_name).name != _name:
        raise ValueError('Unsafe overlay filename')
    _dest.write_bytes(_text.encode('utf-8'))
(bundle/'CONTRACT.json').write_bytes(_derived_contract.encode('utf-8'))
CONTRACT_SHA256 = DERIVED_CONTRACT_SHA
verify_bundle(bundle,CONTRACT_SHA256)
print(json.dumps(dict(event='scheduling_overlay_verified',contract_sha256=CONTRACT_SHA256)),flush=True)
'''.replace('OVERLAY_PAYLOAD',repr(overlays)).replace('DERIVED_CONTRACT_PAYLOAD',repr(contract_text)).replace('DERIVED_CONTRACT_SHA',repr(contract_sha))
    code=replace_once(code,'verify_bundle(bundle, CONTRACT_SHA256)\n',
                       'verify_bundle(bundle, CONTRACT_SHA256)\n'+insertion)
    code=replace_once(code,"smoke_movies = ['44b6_12dfb391','6bba_07e24132'] if RUN_MODE == 'acceptance' else all_movies[:2]",
                       "smoke_movies = ['44b6_12dfb391','44b6_267148e4','6bba_062c8d37','6bba_07e24132'] if RUN_MODE == 'acceptance' else all_movies[:4]")
    ast.parse(code)
    notebook['cells'][0]['source']=['Offline two-T4 scheduling acceptance. Up to two FP32 movie workers per GPU. Same models and predictions logic; no truth or submission.csv.']
    notebook['cells'][1]['source']=code.splitlines(keepends=True)
    filename='trajectory-overlap-acceptance.ipynb'
    (out/filename).write_text(json.dumps(notebook,indent=2))
    metadata=json.loads((base_notebook.parent/'kernel-metadata.json').read_text())
    metadata.update(id='indarkarhana/biohub-trajectory-overlap-acceptance',
                    title='Biohub Trajectory Overlap Acceptance',code_file=filename)
    validate_submission_kernel_metadata(metadata)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    receipt=dict(status='staged_not_launched',kernel=metadata['id'],notebook_sha256=sha(out/filename),
                 contract_sha256=contract_sha,base_contract_sha256=base_contract,
                 overlap_identity_sha256=sha(proof_path),overlay_sha256=sha(out/'overlays.json'),
                 required_gpus=2,maximum_workers=4,acceptance_wall_cap_seconds=3600,
                 no_new_dataset_upload=True,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-build.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
