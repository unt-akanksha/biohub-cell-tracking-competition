"""Label-free, frozen FOCUS support for existing baseline track breaks."""
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from research.focus3d_bridge_rescue import _validate_graph, SCALE_ZYX_UM


def mutual_matches(base, external):
    mapping = {}
    for time in sorted({n['t'] for n in base.values()}):
        b = sorted(k for k, n in base.items() if n['t'] == time)
        f = sorted(k for k, n in external.items() if n['t'] == time)
        if not b or not f:
            continue
        bp = np.asarray([[base[k][a] for a in ('z', 'y', 'x')] for k in b]) * SCALE_ZYX_UM
        fp = np.asarray([[external[k][a] for a in ('z', 'y', 'x')] for k in f]) * SCALE_ZYX_UM
        fd, fi = cKDTree(bp).query(fp, k=2)
        bd, bi = cKDTree(fp).query(bp, k=2)
        for j, key in enumerate(f):
            i = int(fi[j, 0])
            if (fd[j, 0] <= 3.0 and i < len(b) and bi[i, 0] == j
                    and fd[j, 1] - fd[j, 0] > 1e-8
                    and bd[i, 1] - bd[i, 0] > 1e-8):
                mapping[key] = b[i]
    return mapping


def repair(base, external):
    bn = {int(k): v for k, v in base['nodes'].items()}
    fn = {int(k): v for k, v in external['nodes'].items()}
    # Public line fitting can leave slightly negative inherited coordinates.
    # Preserve these exactly; never introduce or conceal a coordinate change.
    _validate_graph(bn, base['edges'], proposal_coordinates=True)
    _validate_graph(fn, external['edges'])
    mapping = mutual_matches(bn, fn)
    bo = Counter(e['source_id'] for e in base['edges'])
    bi = Counter(e['target_id'] for e in base['edges'])
    fo = Counter(e['source_id'] for e in external['edges'])
    proposals = set()
    for e in external['edges']:
        s, t = e['source_id'], e['target_id']
        if s not in mapping or t not in mapping or fo[s] != 1:
            continue
        a, b = mapping[s], mapping[t]
        if not bo[a] and not bi[b] and bn[b]['t'] == bn[a]['t'] + 1:
            proposals.add((a, b))
    outgoing = Counter(a for a, _ in proposals)
    incoming = Counter(b for _, b in proposals)
    accepted = sorted((a, b) for a, b in proposals if outgoing[a] == incoming[b] == 1)
    result = deepcopy(base)
    result['edges'].extend(dict(source_id=a, target_id=b) for a, b in accepted)
    _validate_graph(bn, result['edges'], proposal_coordinates=True)
    assert result['nodes'] == base['nodes']
    assert result['edges'][:len(base['edges'])] == base['edges']
    return result, dict(matched_nodes=len(mapping), proposed_edges=len(proposals),
                        added_edges=len(accepted), added_nodes=0,
                        existing_edges_changed=0, coordinate_changes=0)
