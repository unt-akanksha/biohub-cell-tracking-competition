"""Check pulled Kaggle code/inputs exactly match the launched private stage."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    local=ROOT/'.biohub/staging/biohub-structured-trajectory-acceptance-v1'
    remote=ROOT/'.biohub/cache/trajectory-structured-kaggle-v1-source'
    build=json.loads((ROOT/'reports/experiments/trajectory-structured-kaggle-v1-build.json').read_text())
    settings=json.loads((local/'kernel-metadata.json').read_text())
    observed=json.loads((remote/'kernel-metadata.json').read_text())
    lp=local/settings['code_file'];rp=remote/observed['code_file']
    assert sha(lp)==build['notebook_sha256']
    for key in ('id','title','language','kernel_type','is_private','enable_gpu','enable_tpu','enable_internet',
                'dataset_sources','kernel_sources','competition_sources','model_sources','docker_image','machine_shape'):
        assert observed[key]==settings[key],key
    def cells(path):return [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
    assert cells(lp)==cells(rp)
    result=dict(status='remote_source_verified',kernel=observed['id'],kernel_id=observed['id_no'],
                local_notebook_sha256=sha(lp),remote_notebook_sha256=sha(rp),
                remote_metadata_sha256=sha(remote/'kernel-metadata.json'),
                code_cells_exact=True,private=True,internet_disabled=True,gpu_enabled=True,
                actual_two_t4_execution_still_requires_runtime_result=True)
    (ROOT/'reports/experiments/trajectory-structured-kaggle-v1-remote.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
