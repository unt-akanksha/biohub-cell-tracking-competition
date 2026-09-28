"""Frozen high-confidence native-image repairs; no new nodes or divisions.

Native model probabilities must be computed from image-only inputs. Every row
contains the same nearest-16/20um candidate contract used during training.
"""
from collections import Counter
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree

VOXEL=np.array([1.625,.40625,.40625],np.float32)


def select_model_groups(training,cross):
    """Predeclared CNN-first admission, never target-score mixture optimization."""
    groups=[]
    for embryo in ('6bba','44b6'):
        members={m['family']:m for m in training['members'] if m['embryo']==embryo}
        fold=next((f for f in cross['folds'] if f['source']==embryo),None)
        if fold is None or fold.get('status')!='evaluated_once':continue
        methods=fold.get('methods',{})
        cnn=members.get('resnet3d')
        if not cnn or not cnn['source_gate'] or not methods.get('resnet3d',{}).get('conditional_gate'):continue
        chosen=[cnn]
        transformer=members.get('token_transformer3d')
        if transformer and transformer['source_gate'] and methods.get('token_transformer3d',{}).get('conditional_gate') and methods.get('fixed_equal_mixture',{}).get('conditional_gate'):
            chosen.append(transformer)
        groups.append(chosen)
    return [dict(embryo=m['embryo'],family=m['family'],weights_sha256=m['weights_sha256'],weight=1/len(groups)/len(group))
            for group in groups for m in group]


def frame_candidates(graph,t):
    parents=sorted((n for n in graph['nodes'].values() if n['t']==t),key=lambda n:n['node_id'])
    children=sorted((n for n in graph['nodes'].values() if n['t']==t+1),key=lambda n:n['node_id'])
    pcoords=np.array([[n['z'],n['y'],n['x']] for n in parents],np.float32).reshape(-1,3)*VOXEL
    ccoords=np.array([[n['z'],n['y'],n['x']] for n in children],np.float32).reshape(-1,3)*VOXEL
    ids=np.full((len(children),17),-1,np.int64)
    coords=np.zeros((len(children),17,3),np.float32)
    parent_indices=np.zeros((len(children),16),np.int64)
    tree=cKDTree(pcoords) if len(parents) else None
    for row,child in enumerate(children):
        eligible=np.array(tree.query_ball_point(ccoords[row],20.) if tree else [],np.int64)
        # Stable physical-distance then node-ID order, independent of truth.
        distance=np.linalg.norm(pcoords[eligible]-ccoords[row],axis=1)
        order=np.lexsort((np.array([parents[i]['node_id'] for i in eligible]),distance))[:16]
        selected=eligible[order];n=len(selected)
        ids[row,0]=child['node_id'];ids[row,1:n+1]=[parents[i]['node_id'] for i in selected]
        coords[row,0]=ccoords[row];coords[row,1:n+1]=pcoords[selected];parent_indices[row,:n]=selected
    return dict(ids=ids,coords=coords,valid=ids>=0,parent_indices=parent_indices,
                parent_node_ids=[n['node_id'] for n in parents],child_node_ids=[n['node_id'] for n in children],
                parent_coords=pcoords,child_coords=ccoords)


def propose_repairs(graph,packet,probabilities):
    ids=packet['ids'];valid=packet['valid'];prob=np.asarray(probabilities,np.float64)
    if prob.shape!=(len(ids),17) or not np.isfinite(prob).all() or np.any(prob<0) or np.any(prob>1):
        raise ValueError('Invalid native posterior matrix')
    if not np.allclose(prob.sum(-1),1.,atol=1e-5,rtol=0):raise ValueError('Posterior rows must sum to one')
    if np.any(prob[:,:16][~valid[:,1:]]>1e-7):raise ValueError('Padding cannot carry probability')
    outgoing=Counter(e['source_id'] for e in graph['edges']);incoming={e['target_id']:e['source_id'] for e in graph['edges']}
    proposals=[]
    for row in range(len(ids)):
        choice=int(np.argmax(prob[row]));child=int(ids[row,0]);old=incoming.get(child)
        if choice==16 or prob[row,choice]<.99 or not valid[row,choice+1]:continue
        parent=int(ids[row,choice+1])
        if parent==old or outgoing[parent]!=0:continue
        # Preserve all existing forks and any edge outside the model's candidate
        # context. Only a linear old edge can be displaced by a >=.99 posterior.
        if old is not None and (outgoing[old]!=1 or old not in ids[row,1:]):continue
        if graph['nodes'][str(parent)]['t']+1!=graph['nodes'][str(child)]['t']:raise ValueError('Nonadjacent proposed edge')
        proposals.append(dict(parent=parent,child=child,old_parent=old,probability=float(prob[row,choice]),
                              kind='addition' if old is None else 'rewire'))
    return proposals


def apply_repairs(graph,proposals):
    result=deepcopy(graph);edges={(e['source_id'],e['target_id']) for e in graph['edges']}
    outgoing=Counter(a for a,b in edges);incoming={b:a for a,b in edges};used=set();accepted=[]
    for row in sorted(proposals,key=lambda r:(-r['probability'],r['parent'],r['child'])):
        a,b,old=row['parent'],row['child'],row['old_parent']
        if a in used or outgoing[a]!=0 or incoming.get(b)!=old:continue
        if old is not None:
            if outgoing[old]!=1:continue
            edges.remove((old,b));outgoing[old]-=1
        edges.add((a,b));outgoing[a]+=1;incoming[b]=a;used.add(a);accepted.append(row)
    result['edges']=[dict(source_id=a,target_id=b) for a,b in sorted(edges)]
    return result,accepted
