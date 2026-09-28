"""Sparse-label-safe parent/two-daughter examples at image-derived locations."""
import numpy as np
from research.native_correspondence_data_v2 import VOXEL,match_queries


def legacy_bindings(packet,movie,t):
    nodes={int(n[0]):n for n in movie['nodes']}
    edges=[e for e in movie['edges'] if int(nodes[e[0]][1])==t]
    truth=np.array([nodes[c][2:] for p,c in edges],np.float32).reshape(-1,3)*VOXEL
    query=packet['coords'][:,0]
    distance=np.linalg.norm(query[:,None]-truth[None],axis=-1)
    unique=(distance<=3.25).sum(-1)==1
    nearest=np.argmin(distance,axis=1) if len(edges) else np.zeros(len(query),int)
    count={int(i):int(((nearest==i)&unique).sum()) for i in np.unique(nearest[unique])}
    bindings=[];positions={};omitted=0
    for row,target in enumerate(packet['targets']):
        if target==16:continue
        if not unique[row] or count[int(nearest[row])]!=1:omitted+=1;continue
        parent,child=map(int,edges[int(nearest[row])])
        parent_um=np.array(nodes[parent][2:],np.float32)*VOXEL
        actual_parent=packet['coords'][row,int(target)+1]
        if np.linalg.norm(actual_parent-parent_um)>3.25+1e-5:omitted+=1;continue
        parent_patch=int(packet['ids'][row,int(target)+1]);child_patch=int(packet['ids'][row,0])
        for index,position in ((parent_patch,actual_parent),(child_patch,query[row])):
            if index in positions and not np.allclose(positions[index],position,atol=1e-5):raise ValueError('Patch coordinate identity changed')
            positions[index]=position
        bindings.append(dict(parent=parent,child=child,parent_patch=parent_patch,child_patch=child_patch,true_parent_um=parent_um))
    return bindings,positions,dict(input_groups=len(query),bound_edges=len(bindings),identity_omitted=omitted)


def native_bindings(movie,t,parent_points,child_points):
    parents=[n for n in movie['nodes'] if int(n[1])==t]
    children=[n for n in movie['nodes'] if int(n[1])==t+1]
    p_truth=np.array([n[2:] for n in parents],np.float32).reshape(-1,3)*VOXEL
    c_truth=np.array([n[2:] for n in children],np.float32).reshape(-1,3)*VOXEL
    pm=match_queries(p_truth,parent_points);cm=match_queries(c_truth,child_points)
    p_lookup={int(parents[i][0]):j for i,j in pm.items()};c_lookup={int(children[i][0]):j for i,j in cm.items()}
    truth_by_parent={int(n[0]):p_truth[i] for i,n in enumerate(parents)}
    known={int(c):int(p) for p,c in movie['edges'] if int(p) in truth_by_parent}
    bindings=[];positions={};offset=len(parent_points)
    for child,parent in sorted(known.items()):
        if child not in c_lookup or parent not in p_lookup:continue
        pp=p_lookup[parent];cp=c_lookup[child]+offset
        if np.linalg.norm(parent_points[pp]-child_points[cp-offset])>20.:continue
        bindings.append(dict(parent=parent,child=child,parent_patch=pp,child_patch=cp,true_parent_um=truth_by_parent[parent]))
        positions[pp]=parent_points[pp];positions[cp]=child_points[cp-offset]
    return bindings,positions,dict(input_known_edges=len(known),bound_edges=len(bindings),identity_omitted=len(known)-len(bindings))


def triplets(bindings,positions):
    rows={};ambiguous=0
    for anchor in bindings:
        for other in bindings:
            if anchor['child']==other['child']:continue
            p=anchor['parent_patch'];a=anchor['child_patch'];b=other['child_patch']
            if a==b:raise ValueError('Two daughters cannot share an image detection')
            if np.linalg.norm(positions[p]-positions[b])>20.:continue
            positive=anchor['parent']==other['parent']
            if positive and p!=other['parent_patch']:ambiguous+=1;continue
            if not positive and (p==other['parent_patch'] or np.linalg.norm(positions[p]-other['true_parent_um'])<=7.):
                ambiguous+=1;continue
            first,second=(anchor,other) if a<b else (other,anchor)
            key=(p,min(a,b),max(a,b))
            row=dict(indices=key,label=int(positive),truth_ids=(anchor['parent'],first['child'],second['child']),
                     daughter_parents=(first['parent'],second['parent']))
            if key in rows and rows[key]!=row:raise ValueError('Contradictory triplet label')
            rows[key]=row
    ordered=[rows[k] for k in sorted(rows)]
    if len(ordered)>10000:raise ValueError('Triplet cap exceeded; do not truncate')
    used=sorted({i for r in ordered for i in r['indices']});mapping={old:i for i,old in enumerate(used)}
    packet=dict(triples=np.array([[mapping[i] for i in r['indices']] for r in ordered],np.int64).reshape(-1,3),
                labels=np.array([r['label'] for r in ordered],np.int64),
                coords=np.array([[positions[i] for i in r['indices']] for r in ordered],np.float32).reshape(-1,3,3),
                truth_ids=np.array([r['truth_ids'] for r in ordered],np.int64).reshape(-1,3),
                daughter_parents=np.array([r['daughter_parents'] for r in ordered],np.int64).reshape(-1,2))
    counts=dict(triples=len(ordered),positive=int(packet['labels'].sum()),negative=int((packet['labels']==0).sum()),ambiguous_pairs_omitted=ambiguous)
    return packet,used,counts
