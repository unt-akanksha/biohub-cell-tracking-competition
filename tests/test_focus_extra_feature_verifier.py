import ast
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np
import pytest
from research.focus_cached_pair import validate_pair

ROOT=Path(__file__).resolve().parents[1]
MODULE=runpy.run_path(str(ROOT/'scripts/verify-focus-extra-fit-features.py'))


def packet():
    coords=np.array([[0,1,2,3],[1,2,3,4]],np.float32)
    value=dict(source_frame=np.array(0,np.int64),source_indices=np.array([0],np.int64),
        target_indices=np.array([1],np.int64),source_coords=coords[:1,1:],target_coords=coords[1:,1:],
        backward_um=np.zeros((1,3),np.float32),labels=np.array([0],np.int64))
    for side in ('source','target'):
        for name in ('features','pos'):value[side+'_'+name]=np.zeros((1,32),np.float32)
    movie=dict(role='fitting',windows=[dict(source_frame=0,columns=[0],parent_rows=[0])])
    record=dict(**validate_pair(value,coords),sampling=dict(sampled_nodes=1,coordinates_modified=False,nodes_deleted=False))
    return value,coords,movie,record


def test_exact_packet_passes():
    value,coords,movie,record=packet()
    assert MODULE['verify_packet'](value,coords,movie,0,record)['known_parent']==1


def test_reordered_packet_rejected():
    value,coords,movie,record=packet()
    with pytest.raises(ValueError,match='frame/count'):
        MODULE['verify_packet'](value,coords,movie,1,record)


def test_changed_audit_label_rejected():
    value,coords,movie,record=packet();movie['windows'][0]['parent_rows']=[1]
    with pytest.raises(ValueError,match='frozen audit'):
        MODULE['verify_packet'](value,coords,movie,0,record)


@pytest.mark.parametrize('field',['coordinates_modified','nodes_deleted'])
def test_sampler_mutation_rejected(field):
    value,coords,movie,record=packet();record['sampling'][field]=True
    with pytest.raises(ValueError,match='Sampler'):
        MODULE['verify_packet'](value,coords,movie,0,record)


def test_prelaunch_identity_tamper_rejected(tmp_path):
    build=runpy.run_path(str(ROOT/'scripts/build-focus-extra-fit-features.py'))
    nb,meta=build['build']({'contract':{},'movies':[]})
    notebook=tmp_path/meta['code_file'];notebook.write_bytes(json.dumps(nb).encode())
    metadata=tmp_path/'kernel-metadata.json';metadata.write_bytes(json.dumps(meta).encode())
    identity=dict(run_id=MODULE['RUN'],status='staged_not_launched',declared_budget_seconds=3600,
        notebook_sha256=MODULE['sha'](notebook),metadata_sha256=MODULE['sha'](metadata),
        builder_sha256=MODULE['sha'](ROOT/'scripts/build-focus-extra-fit-features.py'))
    (tmp_path/'staged_identity.json').write_text(json.dumps(identity))
    actual,runtime=MODULE['staged_runtime'](tmp_path)
    assert actual==identity and 'research/focus_extra_feature_scope.py' in runtime
    notebook.write_bytes(notebook.read_bytes()+b' ')
    with pytest.raises(ValueError,match='prelaunch staged'):
        MODULE['staged_runtime'](tmp_path)
