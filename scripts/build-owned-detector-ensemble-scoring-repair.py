"""Short-name metadata repair; preserve exact previously tested CPU notebook."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'kaggle/biohub-owned-detector-ensemble-selection-v1-scoring'
TARGET=ROOT/'kaggle/biohub-owned-detector-ensemble-scoring-v1'
SHA='d7d6ba1823f723f941dcc7ab076a0e92d0e14dd449c27377064b612577bd5f99'


def build():
    meta=json.loads((OLD/'kernel-metadata.json').read_text())
    payload=(OLD/meta['code_file']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=SHA: raise ValueError('Frozen CPU scorer changed')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector Ensemble Scoring v1')
    if len(meta['title'])>50 or len(TARGET.name)>50: raise ValueError('Short Kaggle title/slug required')
    if meta['enable_gpu'] or meta['enable_internet'] or meta['enable_tpu']: raise ValueError('CPU offline repair only')
    return payload,meta


if __name__=='__main__':
    payload,meta=build()
    if TARGET.exists(): raise ValueError('Refuse to overwrite CPU repair staging')
    TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_bytes(payload)
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
