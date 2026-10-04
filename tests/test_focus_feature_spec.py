import copy
import numpy as np
import pytest
from research.focus_feature_spec import sparse_windows,validate_spec


def fixture():
    coords=np.array([[t,1,2,3] for t in range(100)],dtype=np.float32)
    rows=[dict(source_frame=t,role='fitting',source_indices=[t],target_indices=[t+1],null_index=1,labels=[-1 if t%3==0 else t%2]) for t in range(99)]
    return coords,dict(role='fitting',frames=list(range(100)),rows=rows)


def test_conversion_preserves_unknown_and_null():
    coords,audit=fixture();windows=sparse_windows(audit,coords,'fitting')
    for row,w in zip(audit['rows'],windows):
        labels=np.full(w['target_count'],-1)
        labels[w['columns']]=w['parent_rows']
        assert labels.tolist()==row['labels']


@pytest.mark.parametrize('mutation',['role','identity','labels','coverage'])
def test_bad_audit_rejected(mutation):
    coords,audit=fixture()
    if mutation=='role':audit['rows'][0]['role']='diagnostic'
    if mutation=='identity':audit['rows'][0]['source_indices']=[1]
    if mutation=='labels':audit['rows'][0]['labels']=[2]
    if mutation=='coverage':audit['rows'].pop()
    with pytest.raises(ValueError):sparse_windows(audit,coords,'fitting')


def test_spec_rejects_role_swap_and_duplicate_columns():
    coords,audit=fixture();windows=sparse_windows(audit,coords,'fitting')
    contract=dict(replay_stems=['r'],fitting_stems=['f'],diagnostic_stems=['d'])
    movies=[dict(stem='r',role='replay',raw_sha256='a'*64,probe_sha256='b'*64,normalized_image_sha256=['c'*64]*2)]
    movies.extend(dict(stem=s,role=r,raw_sha256='a'*64,labels_sha256='b'*64,windows=copy.deepcopy(windows)) for s,r in [('f','fitting'),('d','diagnostic')])
    spec=dict(contract=contract,movies=movies);validate_spec(spec,contract)
    bad=copy.deepcopy(spec);bad['movies'][1]['role']='diagnostic'
    with pytest.raises(ValueError):validate_spec(bad,contract)
    bad=copy.deepcopy(spec);bad['movies'][1]['windows'][0].update(columns=[0,0],parent_rows=[0,0])
    with pytest.raises(ValueError):validate_spec(bad,contract)
