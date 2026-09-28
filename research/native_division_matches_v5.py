"""Add isolated matches without displacing any strict v2 image/GT assignment.

Candidate training-data policy only. Never use GT to generate inference points.
"""
from collections import Counter
import numpy as np
from research.native_correspondence_data_v2 import match_queries


def isolated_additive_matches(truth_um,image_um):
    strict=match_queries(truth_um,image_um)
    if not len(truth_um) or not len(image_um):return strict
    distances=np.linalg.norm(truth_um[:,None]-image_um[None],axis=-1)
    used=set(strict.values());candidates={}
    for i in range(len(truth_um)):
        if i in strict:continue
        order=np.argsort(distances[i],kind='stable');nearest=int(order[0])
        if nearest in used or distances[i,nearest]>7:continue
        if len(order)>1 and distances[i,order[1]]-distances[i,nearest]<3.25:continue
        candidates[i]=nearest
    ownership=Counter(candidates.values())
    result={**strict,**{i:j for i,j in candidates.items() if ownership[j]==1}}
    if len(result.values())!=len(set(result.values())) or any(result[i]!=j for i,j in strict.items()):
        raise ValueError('Additive matching must preserve all strict assignments')
    return result
