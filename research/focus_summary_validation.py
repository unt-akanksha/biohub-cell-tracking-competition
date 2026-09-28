"""Exact known-label identity checks for compact parent-presence evidence."""
import numpy as np
from research.focus_parent_presence import FEATURES,metrics

FIELDS={'context','offset','present','conditional_nll','conditional_max','correct_if_present','target_indices','source_frame'}


def validate_summary(arrays,record,labels):
    n=record['rows']
    if (n<=0 or set(arrays)!=FIELDS or arrays['context'].shape!=(n,len(FEATURES))
        or any(v.shape!=(n,) for k,v in arrays.items() if k!='context')
        or any(not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('Finite complete nonempty presence summary required')
    for key in ('present','correct_if_present','target_indices','source_frame'):
        if arrays[key].dtype!=np.int64:raise ValueError('Integer label/identity arrays required')
    ids=[];present=[];frames=[]
    for row in labels['rows']:
        if len(row['target_indices'])!=len(row['labels']):raise ValueError('Audited target/label lengths differ')
        for index,label in zip(row['target_indices'],row['labels']):
            if label>=0:
                ids.append(index);present.append(int(label<row['null_index']));frames.append(row['source_frame'])
    if not all(np.array_equal(arrays[k],v) for k,v in [('target_indices',ids),('present',present),('source_frame',frames)]):
        raise ValueError('Known-label/global-target identities differ from complete sparse audit')
    if (not set(arrays['correct_if_present'])<={0,1}
        or ((arrays['present']==0)&(arrays['correct_if_present']!=0)).any()):
        raise ValueError('Invalid parent correctness flags')
    replay=metrics(arrays);old=record['baseline']
    if (any(replay[k]!=old[k] for k in ('known_parent','known_absent','correct_parent','correct_absent'))
        or abs(replay['nll']-old['nll'])>2e-6):raise ValueError('Summary baseline replay differs')
    return replay
