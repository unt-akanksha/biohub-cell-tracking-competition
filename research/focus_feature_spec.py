"""Fail-closed scope and sparse-label conversion for a frozen feature cache."""
import numpy as np


def sparse_windows(audit, coords, role):
    if audit['role'] != role or audit['frames'] != list(range(100)):
        raise ValueError('Complete fixed-role label audit required')
    if [r['source_frame'] for r in audit['rows']] != list(range(99)):
        raise ValueError('All adjacent windows in original order required')
    windows=[]
    for row in audit['rows']:
        t=row['source_frame'];source=np.flatnonzero(coords[:,0]==t);target=np.flatnonzero(coords[:,0]==t+1)
        labels=np.asarray(row['labels'])
        if (row['role']!=role or row['source_indices']!=source.tolist() or row['target_indices']!=target.tolist()
            or row['null_index']!=len(source) or labels.shape!=(len(target),)
            or (labels.size and not np.issubdtype(labels.dtype,np.integer))
            or (labels< -1).any() or (labels>len(source)).any()):
            raise ValueError('Labels must retain exact raw proposal identities')
        columns=np.flatnonzero(labels>=0)
        windows.append(dict(source_frame=t,source_count=len(source),target_count=len(target),
                            columns=columns.tolist(),parent_rows=labels[columns].tolist()))
    return windows


def validate_spec(spec, contract):
    if spec['contract']!=contract:raise ValueError('Frozen training-only scope required')
    expected=[(s,'replay') for s in contract['replay_stems']]+[(s,'fitting') for s in contract['fitting_stems']]+[(s,'diagnostic') for s in contract['diagnostic_stems']]
    if [(r['stem'],r['role']) for r in spec['movies']]!=expected:
        raise ValueError('Exact replay-first and four/four movie roles required')
    for row in spec['movies']:
        if len(row['raw_sha256'])!=64:raise ValueError('Raw artifact digest required')
        if row['role']=='replay':
            if len(row['normalized_image_sha256'])!=2 or len(row['probe_sha256'])!=64:
                raise ValueError('Two exact earlier image/head replay identities required')
        else:
            if len(row['labels_sha256'])!=64 or [w['source_frame'] for w in row['windows']]!=list(range(99)):
                raise ValueError('Complete audited label provenance required')
            for window in row['windows']:
                columns=window['columns'];parents=window['parent_rows'];ns=window['source_count'];nt=window['target_count']
                if (not 0<=ns<=2048 or not 0<=nt<=2048 or len(columns)!=len(parents)
                    or columns!=sorted(set(columns)) or any(type(i)!=int or not 0<=i<nt for i in columns)
                    or any(type(i)!=int or not 0<=i<=ns for i in parents)):
                    raise ValueError('Bounded unique sparse targets with explicit null class required')

