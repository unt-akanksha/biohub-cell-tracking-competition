"""Joint high-confidence identity cycles and persistent two-daughter events."""
from collections import defaultdict
from copy import deepcopy
import numpy as np


def links(graph):
    outgoing=defaultdict(list);incoming={}
    for edge in graph['edges']:
        outgoing[edge['source_id']].append(edge['target_id']);incoming[edge['target_id']]=edge['source_id']
    return incoming,outgoing


def choices(ids,probabilities):
    ids=np.asarray(ids);prob=np.asarray(probabilities)
    if ids.ndim!=2 or ids.shape[1]!=17 or prob.shape!=ids.shape or len(set(ids[:,0]))!=len(ids):raise ValueError('Unique complete query rows required')
    if not np.isfinite(prob).all() or np.any(prob<0) or not np.allclose(prob.sum(-1),1,atol=1e-5):raise ValueError('Normalized finite probabilities required')
    best=prob.argmax(-1);result={}
    for row,col in enumerate(best):
        if col==16 or prob[row,col]<.99:continue
        parent=int(ids[row,col+1]);child=int(ids[row,0])
        if parent<0:raise ValueError('Padded best parent')
        result[child]=(parent,float(prob[row,col]))
    return result


def persistent(node,incoming,outgoing,direction):
    """Three observed consecutive frames, with no intervening branch."""
    path=[node]
    for _ in range(2):
        if direction=='past':
            previous=incoming.get(path[-1])
            if previous is None or len(outgoing[previous])!=1:return False
            path.append(previous)
        else:
            children=outgoing[path[-1]]
            if len(children)!=1:return False
            path.append(children[0])
    return True


def joint_repair(graph,ids,probabilities):
    selected=choices(ids,probabilities);incoming,outgoing=links(graph)
    edges={(e['source_id'],e['target_id']) for e in graph['edges']}
    arrows={};edge_child={}
    for child,(parent,confidence) in selected.items():
        old=incoming.get(child)
        if old is None or old==parent or len(outgoing[old])!=1 or len(outgoing[parent])!=1:continue
        if graph['nodes'][str(parent)]['t']+1!=graph['nodes'][str(child)]['t']:raise ValueError('Nonadjacent model choice')
        arrows[old]=parent;edge_child[old]=child
    visited=set();events=[]
    for first in sorted(arrows):
        chain=[];where={};current=first
        while current in arrows and current not in visited and current not in where:
            where[current]=len(chain);chain.append(current);current=arrows[current]
        if current in where:
            cycle=chain[where[current]:]
            if len(cycle)<2:raise ValueError('Unexpected identity cycle')
            changes=[]
            for old in cycle:
                child=edge_child[old];parent=arrows[old]
                edges.remove((old,child));changes.append(dict(old_parent=old,parent=parent,child=child,probability=selected[child][1]))
            edges.update((r['parent'],r['child']) for r in changes)
            events.append(dict(kind='joint_identity_cycle',changes=changes))
        visited.update(chain)
    intermediate=dict(nodes=graph['nodes'],edges=[dict(source_id=a,target_id=b) for a,b in sorted(edges)])
    incoming,outgoing=links(intermediate);used=set()
    for child,(parent,confidence) in sorted(selected.items(),key=lambda row:(-row[1][1],row[1][0],row[0])):
        if child in incoming or parent in used or len(outgoing[parent])!=1:continue
        sibling=outgoing[parent][0]
        if sibling not in selected or selected[sibling][0]!=parent:continue
        if graph['nodes'][str(parent)]['t']+1!=graph['nodes'][str(child)]['t']:raise ValueError('Nonadjacent daughter')
        if not persistent(parent,incoming,outgoing,'past'):continue
        if not persistent(child,incoming,outgoing,'future') or not persistent(sibling,incoming,outgoing,'future'):continue
        edges.add((parent,child));incoming[child]=parent;outgoing[parent].append(child);used.add(parent)
        events.append(dict(kind='persistent_division',parent=parent,existing_child=sibling,new_child=child,
                           new_child_probability=confidence,existing_child_probability=selected[sibling][1]))
    result=deepcopy(graph);result['edges']=[dict(source_id=a,target_id=b) for a,b in sorted(edges)]
    return result,events
