"""Independent graph invariants for event-stage deployment acceptance."""
from collections import Counter

from research.trajectory_runtime_v1 import validate_graph


def pairs(graph):
    edges={(int(e['source_id']),int(e['target_id'])) for e in graph['edges']}
    if len(edges)!=len(graph['edges']):raise ValueError('Duplicate graph edges')
    return edges


def check_event_stage(initial,baseline,candidate,details,frames=100):
    validate_graph(baseline,frames);validate_graph(candidate,frames)
    if baseline['nodes']!=candidate['nodes']:raise ValueError('Event stage changed cells or coordinates')
    old,new=pairs(baseline),pairs(candidate);added,removed=new-old,old-new
    nodes=baseline['nodes'];observed=set(map(int,initial['nodes']))&set(map(int,nodes))
    outgoing=Counter(a for a,b in old);protected=set()
    for a,b in old:
        if a not in observed or b not in observed or outgoing[a]==2 or nodes[str(b)]['t']-nodes[str(a)]['t']!=1:
            protected.update((a,b))
    incident=lambda edges:{(a,b) for a,b in edges if a in protected or b in protected}
    if incident(old)!=incident(new):raise ValueError('Protected division, synthetic, or gap incidence changed')
    if max(Counter(b for a,b in new).values(),default=0)>1 or max(Counter(a for a,b in new).values(),default=0)>2:
        raise ValueError('Lineage degree violation')
    if any(a not in observed or b not in observed or nodes[str(b)]['t']-nodes[str(a)]['t']!=1 for a,b in added|removed):
        raise ValueError('Changed edge is outside observed consecutive scope')
    if len(added)!=details['added_edges'] or len(removed)!=details['removed_edges']:
        raise ValueError('Recorded edit counts differ from actual graph')
    if not details['nodes_and_coordinates_unchanged']:raise ValueError('Missing node preservation claim')
    rows=details['frames'];times=[int(r['t']) for r in rows]
    if len(times)!=len(set(times)) or len(times)!=details['processed_frames'] or any(t<1 or t>=frames for t in times):
        raise ValueError('Invalid processed transition inventory')
    if details['solver_fallbacks']!=sum(bool(r['fallback']) for r in rows):raise ValueError('Fallback count differs')
    for row in rows:
        t=int(row['t']);a=sum(nodes[str(b)]['t']==t for _,b in added);r=sum(nodes[str(b)]['t']==t for _,b in removed)
        if a!=row.get('added_edges',0) or r!=row.get('removed_edges',0):raise ValueError('Per-transition edits differ')
        if row['fallback'] and (a or r):raise ValueError('Fallback transition changed')
    if any(nodes[str(b)]['t'] not in times for _,b in added|removed):raise ValueError('Unprocessed transition changed')
    return dict(added_edges=len(added),removed_edges=len(removed),nodes_unchanged=True,
        protected_incidence_unchanged=True,unprocessed_and_fallback_edges_preserved=True)
