"""Verify immutable smoke notebook/source hashes and real output receipts."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-ensemble-probe-v1'


def summarize():
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    folder=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/owned_detector_ensemble_probe'
    nb=json.loads(notebook.read_text())
    source=''.join(nb['cells'][1]['source'])
    for name,file in (('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        bundle=ast.literal_eval(node.value)
        if json.loads((folder/file).read_text())!={k:hashlib.sha256(v.encode()).hexdigest() for k,v in bundle.items()}:
            raise ValueError('Executed bundle differs from frozen launch')
    terminal=json.loads((folder/'launcher_terminal.json').read_text())
    if terminal['status']!='completed' or terminal['run_id']!=RUN or not 0<terminal['elapsed_seconds']<=3600:
        raise ValueError('Completed bounded smoke required')
    result=folder/'outputs/result.json'
    report=dict(status='verified_owned_detector_ensemble_probe_not_selection',
        result=json.loads(result.read_text()),terminal=terminal,
        source_sha256=dict(notebook=hashlib.sha256(notebook.read_bytes()).hexdigest(),
                           result=hashlib.sha256(result.read_bytes()).hexdigest()))
    payload=json.dumps(report,indent=2)
    runpy.run_path(str(ROOT/'research/owned_detector_ensemble_contract.py'))['receipt'](
        payload.encode(),(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists(): raise ValueError('Refuse to overwrite verified ensemble report')
    target.write_text(payload)
    print(json.dumps(report,indent=2))


if __name__=='__main__': summarize()
