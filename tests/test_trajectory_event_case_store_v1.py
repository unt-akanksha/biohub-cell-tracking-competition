import hashlib
import numpy as np
import pytest

from research.trajectory_event_assignment_v1 import problem,prepare
from research.trajectory_event_case_store_v1 import CaseStore


def write_case(root,name,corrupt_allowed=False):
    p=problem([(0,-1,-1),(0,0,-1),(-1,0,-1)],1,1)
    x=np.eye(3);targets=np.array([0]);safe=np.zeros((3,2),bool)
    case,_=prepare(p,x,targets,safe)
    allowed=case['allowed'].copy()
    if corrupt_allowed:allowed[:]=True
    path=root/name
    np.savez_compressed(path,options=p['options'],features=x,targets=targets,safe=safe,
        nparents=1,nchildren=1,incumbent=np.array([0,1,0]),allowed=allowed,margin=case['margin'])
    return dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n_constraints=1)


def test_exact_replay_and_bounded_cache(tmp_path):
    records=[write_case(tmp_path,str(i)+'.npz') for i in range(3)]
    store=CaseStore(tmp_path,records,cache_size=1)
    assert len(store)==3 and store[0]['n_constraints']==1
    assert store[1]['n_constraints']==1 and len(store.cache)==1 and 0 not in store.cache
    assert store[-1]['n_constraints']==1
    with pytest.raises(IndexError):store[3]


def test_modified_data_and_false_allowed_mask_are_rejected(tmp_path):
    row=write_case(tmp_path,'good.npz');row['sha256']='0'*64
    with pytest.raises(ValueError,match='hash changed'):CaseStore(tmp_path,[row])[0]
    row=write_case(tmp_path,'bad.npz',corrupt_allowed=True)
    with pytest.raises(ValueError,match='exact reconstruction'):CaseStore(tmp_path,[row])[0]


def test_outside_store_and_duplicate_entries_rejected(tmp_path):
    row=write_case(tmp_path,'case.npz')
    with pytest.raises(ValueError,match='Duplicate'):CaseStore(tmp_path,[row,row])
    child=tmp_path/'child';child.mkdir();row['path']='../case.npz'
    with pytest.raises(ValueError,match='escapes'):CaseStore(child,[row])[0]
