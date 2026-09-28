"""Partial source supervision for atomic corrections, never inference features."""
from collections import Counter
import numpy as np


def labels(parts,baseline,candidate,groups,targets,safe):
    targets=np.asarray(targets);safe=np.asarray(safe)
    if (targets.shape!=(len(groups['children']),) or targets.dtype.kind not in 'iu'
            or np.any(targets < -1) or safe.shape!=(len(groups['parents']),) or safe.dtype.kind!='b'):
        raise ValueError('Invalid sparse supervision schema')
    child_index={int(c):i for i,c in enumerate(groups['children'])}
    child=np.repeat(groups['children'],np.diff(groups['offsets']))
    pairs=list(zip(map(int,groups['parents']),map(int,child)))
    edge_index={p:i for i,p in enumerate(pairs)}
    if len(child_index)!=len(groups['children']) or len(edge_index)!=len(pairs):
        raise ValueError('Duplicate supervision groups')
    def parents(graph):
        result={int(e['target_id']):int(e['source_id']) for e in graph['edges']}
        if len(result)!=len(graph['edges']):raise ValueError('Multiple parents')
        return result
    old,new=parents(baseline),parents(candidate)
    result=[]
    for part in parts:
        counts=Counter()
        for c in sorted({int(b) for _,b in (*part['removed'],*part['added'])}):
            if c not in child_index:raise ValueError('Changed child lacks source group')
            expected=int(targets[child_index[c]])
            if expected<0:
                counts['unknown_children']+=1
                continue
            if (expected,c) not in edge_index:raise ValueError('Known parent absent from source candidates')
            before,after=old.get(c),new.get(c)
            def known_wrong_or_correct(p):
                if p is None or p==expected:return True
                if (p,c) not in edge_index:raise ValueError('Changed parent lacks source candidate')
                return bool(safe[edge_index[p,c]])
            if not known_wrong_or_correct(before) or not known_wrong_or_correct(after):
                counts['ambiguous_children']+=1
                continue
            counts['known_children']+=1
            delta=int(after==expected)-int(before==expected)
            counts['delta_correct_links']+=delta
            counts['improved_children']+=int(delta>0)
            counts['worsened_children']+=int(delta<0)
        delta=counts['delta_correct_links']
        result.append(dict(counts,label=(1 if delta>0 else 0 if delta<0 else -1)))
    return result
