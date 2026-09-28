"""Recover only genuine, strong, short motion components omitted by output pruning."""
import copy
import numpy as np


def recover(final, reference, motion_edges, config, smooth, pre_filter_node_count):
    nodes={int(k):v for k,v in reference['nodes'].items()}
    kept={int(k) for k in final['nodes']}
    adjacency={i:set() for i in nodes}; incoming={};outgoing={}
    for edge in motion_edges:
        a,b=int(edge['source_id']),int(edge['target_id'])
        if a not in nodes or b not in nodes or nodes[b]['t']!=nodes[a]['t']+1:
            raise ValueError('Invalid original motion edge')
        if a in outgoing or b in incoming:raise ValueError('Motion graph must be one-to-one')
        outgoing[a]=edge;incoming[b]=edge
        adjacency[a].add(b);adjacency[b].add(a)
    visited=set(); proposals=[];tested=0
    for start in sorted(nodes):
        if start in visited:continue
        stack=[start];members=[]
        while stack:
            ident=stack.pop()
            if ident in visited:continue
            visited.add(ident);members.append(ident);stack.extend(adjacency[ident]-visited)
        members.sort()
        if set(members)&kept:continue
        if not config['SHORT_TRACK_RESCUE_MIN_LEN']<=len(members)<config['OUTPUT_MIN_TRACK_LEN']:
            continue
        tested+=1;member_set=set(members)
        edges=[outgoing[i] for i in members if i in outgoing and int(outgoing[i]['target_id']) in member_set]
        if len(edges)!=len(members)-1:raise ValueError('Short component is not a complete linear path')
        probs=np.asarray([float(e.get('edge_prob',0.)) for e in edges])
        dists=np.asarray([float(e['distance_um']) for e in edges])
        if not np.isfinite(probs).all() or not np.isfinite(dists).all():raise ValueError('Nonfinite component evidence')
        mean_prob=float(probs.mean());mean_dist=float(dists.mean())
        if mean_prob<config['SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB'] or mean_dist>config['SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM']:
            continue
        score=mean_prob-.02*mean_dist+.004*len(members)
        proposals.append((score,len(members),mean_prob,min(members),members,edges,mean_dist))
    proposals.sort(key=lambda row:row[:4],reverse=True)
    budget=min(config['SHORT_TRACK_RESCUE_MAX_NODES_ABS'],max(0,int(round(
        pre_filter_node_count*config['SHORT_TRACK_RESCUE_MAX_NODES_FRAC']))))
    result=copy.deepcopy(final); added=0;records=[]
    for _,size,prob,_,members,edges,distance in proposals:
        if added+size>budget:continue
        restored={i:copy.deepcopy(nodes[i]) for i in members}
        smooth_stats={'linefit_skipped_nodes':0,'linefit_smoothed_nodes':0}
        restored=smooth(restored,copy.deepcopy(edges),smooth_stats)
        for ident,node in restored.items():
            if str(ident) in result['nodes']:raise ValueError('Recovery overwrote a retained node')
            row={k:int(node[k]) for k in ('node_id','t')}
            for axis,bound in zip(('z','y','x'),(64,256,256)):
                value=max(0,int(round(float(node[axis]))))
                if value>=bound:raise ValueError('Restored point out of image')
                row[axis]=value
            result['nodes'][str(ident)]=row
        result['edges'].extend({k:int(e[k]) for k in ('source_id','target_id')} for e in edges)
        added+=size
        records.append(dict(node_ids=members,mean_edge_probability=prob,mean_edge_distance_um=distance))
    if any(result['nodes'][i]!=n for i,n in final['nodes'].items()):raise ValueError('Changed an existing node')
    if result['edges'][:len(final['edges'])]!=final['edges']:raise ValueError('Changed an existing edge')
    return result,dict(tested_short_components=tested,eligible_components=len(proposals),
        recovered_components=len(records),added_nodes=added,added_edges=sum(len(r['node_ids'])-1 for r in records),
        node_budget=budget,components=records,existing_output_unchanged=True,
        invented_detections=False,ground_truth_used=False,authorized_for_submission=False)
