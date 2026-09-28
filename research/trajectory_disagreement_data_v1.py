"""Label-free real decision features and conservative sparse-GT supervision."""
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree
from research.native_correspondence_data_v2 import match_queries

SCALE=np.array([1.625,.40625,.40625],np.float64)
FEATURES=('neural_probability','motion_probability','motion_probability_known','probability_delta',
          'neural_raw_distance','motion_raw_distance','neural_final_distance','motion_final_distance',
          'neural_history_residual','motion_history_residual','neural_history_known','motion_history_known',
          'neural_future_residual','motion_future_residual','future_known','future_observed',
          'child_localization_shift','neural_parent_shift','motion_parent_shift',
          'neural_parent_outdegree','motion_parent_outdegree','child_outdegree',
          'neural_parent_indegree','motion_parent_indegree','local_parent_count_10um',
          'initial_neural_parent_outdegree','initial_motion_parent_outdegree','initial_child_outdegree')


def positions(nodes):
    result={int(i):np.array([n[a] for a in ('z','y','x')],np.float64)*SCALE for i,n in nodes.items()}
    if any(not np.isfinite(p).all() for p in result.values()):raise ValueError('Nonfinite geometry')
    return result


def adjacency(edges):
    incoming=defaultdict(list);outgoing=defaultdict(list)
    for e in edges:
        a,b=int(e['source_id']),int(e['target_id'])
        incoming[b].append(a);outgoing[a].append(b)
    return incoming,outgoing


def extract(initial,final,raw_coords,raw_edges):
    """No GT parameter; prospective choices come only from unchanged predictions."""
    initial_nodes={int(k):v for k,v in initial['nodes'].items()}
    final_nodes={int(k):v for k,v in final['nodes'].items()}
    raw_coords=np.asarray(raw_coords);raw_edges=np.asarray(raw_edges)
    if raw_coords.ndim!=2 or raw_coords.shape[1]!=4 or raw_edges.ndim!=2 or raw_edges.shape[1]!=4:
        raise ValueError('Pinned predictor coordinate/edge schema required')
    if not np.isfinite(raw_coords).all() or not np.isfinite(raw_edges).all():raise ValueError('Nonfinite raw inputs')
    for ident,n in initial_nodes.items():
        if not 0<=ident<len(raw_coords) or not np.array_equal(raw_coords[ident],[n[k] for k in ('t','z','y','x')]):
            raise ValueError('Original ILP IDs no longer identify raw detector coordinates')
    probabilities={}
    for a,b,p,d in raw_edges:
        if a!=int(a) or b!=int(b) or not 0<=a<len(raw_coords) or not 0<=b<len(raw_coords) or not 0<=p<=1 or d<0:
            raise ValueError('Invalid raw association')
        key=int(a),int(b)
        if key in probabilities:raise ValueError('Duplicate raw association')
        probabilities[key]=float(p)
    initial_in,initial_out=adjacency(initial['edges']);final_in,final_out=adjacency(final['edges'])
    initial_p=positions(initial_nodes);final_p=positions(final_nodes)
    initial_probs={(int(e['source_id']),int(e['target_id'])):float(e['edge_prob']) for e in initial['edges']}
    by_t=defaultdict(list)
    for i,n in final_nodes.items():by_t[int(n['t'])].append(i)
    trees={t:cKDTree(np.stack([final_p[i] for i in ids])) for t,ids in by_t.items()}
    counts=Counter();records=[];features=[]
    for child in sorted(final_nodes):
        if len(initial_in[child])!=1 or len(final_in[child])!=1:continue
        neural,motion=initial_in[child][0],final_in[child][0]
        if neural==motion:counts['parent_agreement']+=1;continue
        counts['parent_disagreement']+=1
        if not {child,neural,motion}<=initial_nodes.keys() or not {neural,motion}<=final_nodes.keys():
            counts['nonshared_observed_endpoint']+=1;continue
        t=int(final_nodes[child]['t'])
        if any(int(final_nodes[p]['t'])!=t-1 for p in (neural,motion)):raise ValueError('Nonconsecutive decision')
        pn=initial_probs[(neural,child)]
        if not 0<=pn<=1:raise ValueError('Invalid initial edge probability')
        known=(motion,child) in probabilities;pm=probabilities.get((motion,child),.5)
        history=[];history_known=[]
        for parent in (neural,motion):
            available=len(final_in[parent])==1
            prev=final_in[parent][0] if available else None
            history.append(float(np.linalg.norm(final_p[child]-2*final_p[parent]+final_p[prev])) if available else 0.)
            history_known.append(float(available))
        future_known=len(final_out[child])==1
        future=final_out[child][0] if future_known else None
        future_residual=[float(np.linalg.norm(final_p[future]-2*final_p[child]+final_p[p])) if future_known else 0. for p in (neural,motion)]
        raw_dist=[float(np.linalg.norm(initial_p[child]-initial_p[p])) for p in (neural,motion)]
        final_dist=[float(np.linalg.norm(final_p[child]-final_p[p])) for p in (neural,motion)]
        shifts=[float(np.linalg.norm(initial_p[p]-final_p[p])) for p in (child,neural,motion)]
        motif='occupied_parent'
        if len(final_out[neural])==0:motif='free_parent'
        elif len(final_out[neural])==1:
            other=final_out[neural][0]
            if initial_in[other]==[motion] and len(final_out[motion])==1:motif='closed_two_parent_swap'
        if any(len(out[p])>1 for out in (initial_out,final_out) for p in (neural,motion,child)):motif='division_adjacent'
        vector=[pn,pm,float(known),pn-pm,*raw_dist,*final_dist,*history,*history_known,*future_residual,
                float(future_known),float(future in initial_nodes if future_known else False),*shifts,
                len(final_out[neural]),len(final_out[motion]),len(final_out[child]),
                len(final_in[neural]),len(final_in[motion]),len(trees[t-1].query_ball_point(final_p[child],10.)),
                len(initial_out[neural]),len(initial_out[motion]),len(initial_out[child])]
        if len(vector)!=len(FEATURES) or not np.isfinite(vector).all():raise ValueError('Feature schema drift')
        records.append(dict(child=child,neural_parent=neural,motion_parent=motion,t=t,motif=motif))
        features.append(vector);counts['eligible_observed_disagreement']+=1;counts['motif_'+motif]+=1
    return records,np.asarray(features,dtype=np.float64).reshape(-1,len(FEATURES)),dict(counts)


def supervision(records,final_nodes,truth_nodes,truth_edges):
    """Unknown children are omitted; near alternatives are ambiguous, never negative."""
    final_nodes={int(k):v for k,v in final_nodes.items()};truth_nodes={int(k):v for k,v in truth_nodes.items()}
    fp=positions(final_nodes);gp=positions(truth_nodes);det_to_gt={};gt_to_det={}
    for t in sorted({int(n['t']) for n in final_nodes.values()}):
        detections=sorted(i for i,n in final_nodes.items() if int(n['t'])==t)
        annotations=sorted(i for i,n in truth_nodes.items() if int(n['t'])==t)
        if not annotations:continue
        matches=match_queries(np.stack([gp[i] for i in annotations]),np.stack([fp[i] for i in detections]))
        for i,j in matches.items():det_to_gt[detections[j]]=annotations[i];gt_to_det[annotations[i]]=detections[j]
    parents=defaultdict(list)
    for a,b in truth_edges:parents[int(b)].append(int(a))
    labels=np.full(len(records),-1,np.int64);reasons=[];counts=Counter()
    for i,r in enumerate(records):
        child=r['child'];gt=det_to_gt.get(child)
        if gt is None:reason='unannotated_or_unmatched_child'
        elif len(parents[gt])!=1:reason='parent_annotation_unavailable'
        else:
            parent=parents[gt][0]
            if int(truth_nodes[parent]['t'])!=int(truth_nodes[gt]['t'])-1:reason='nonconsecutive_annotation'
            else:
                choices=(r['neural_parent'],r['motion_parent'])
                distances=[float(np.linalg.norm(fp[p]-gp[parent])) for p in choices]
                positive=[gt_to_det.get(parent)==p and distances[j]<=3.25 for j,p in enumerate(choices)]
                if positive[0] and distances[1]>7.:labels[i]=0;reason='neural_parent_correct'
                elif positive[1] and distances[0]>7.:labels[i]=1;reason='motion_parent_correct'
                elif min(distances)>7.:labels[i]=2;reason='neither_parent_correct'
                else:reason='ambiguous_parent_alternative'
        counts[reason]+=1;reasons.append(reason)
    assert sum(counts.values())==len(records)
    return labels,reasons,dict(counts)
