"""Path-only bootstrap repair; preserve already uploaded/validated model runtime."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha

def main():
    prior=ROOT/'.biohub/staging/biohub-trajectory-motion-acceptance-v2'
    out=ROOT/'.biohub/staging/biohub-trajectory-motion-acceptance-v3'
    out.mkdir(exist_ok=False)
    build=json.loads((ROOT/'reports/experiments/trajectory-kaggle-runtime-v2-build.json').read_text())
    code=(ROOT/'scripts/trajectory-kaggle-bootstrap-v1.py').read_text()
    code=code.replace('__ARCHIVE_SHA256__',build['archive_sha256']).replace('__CONTRACT_SHA256__',build['contract_sha256']).replace('__RUN_MODE__','acceptance')
    ast.parse(code)
    if "glob('**/" in code:raise ValueError('No recursive competition input discovery allowed')
    notebook=json.loads((prior/'trajectory-motion-acceptance.ipynb').read_text())
    notebook['cells'][0]['id']='runtime-description'
    notebook['cells'][1]['id']='offline-acceptance'
    notebook['cells'][1]['source']=code.splitlines(keepends=True)
    (out/'trajectory-motion-acceptance.ipynb').write_text(json.dumps(notebook,indent=2))
    (out/'kernel-metadata.json').write_bytes((prior/'kernel-metadata.json').read_bytes())
    receipt=dict(status='staged_not_launched',bootstrap_revision=3,
                 contract_sha256=build['contract_sha256'],archive_sha256=build['archive_sha256'],
                 notebook_sha256=sha(out/'trajectory-motion-acceptance.ipynb'),
                 runtime_changed=False,model_or_inference_changed=False,
                 cause='Kaggle auto-expands dataset ZIP; recursive archive search had no match',
                 submission_performed=False)
    (ROOT/'reports/experiments/trajectory-kaggle-bootstrap-v3-build.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
