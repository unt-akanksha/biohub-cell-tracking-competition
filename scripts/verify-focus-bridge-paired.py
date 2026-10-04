"""Verify saved bridge graph mutations and recompute reported metric arithmetic.

This verifies artifacts and aggregate arithmetic, not the GT matching itself.
Official scorer execution remains evidenced by the pinned notebook and run log.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

STEMS = {'44b6_81c256f0','44b6_24264f12','6bba_f1fde7e0','6bba_23af9eeb'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def aggregate(rows):
    weights = [r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in rows]
    if not sum(weights):
        raise ValueError('No scored edge mass')
    adjusted = sum(w*r['adj_edge_jaccard'] for w,r in zip(weights,rows))/sum(weights)
    tp, fp, fn = [sum(r['division_'+k] for r in rows) for k in ('tp','fp','fn')]
    division = tp/(tp+fp+fn) if tp+fp+fn else float('nan')
    return {'adj_edge_jaccard': adjusted, 'division_jaccard': division,
            'score': adjusted + 0.1*division if tp+fp+fn else adjusted}


def check_mutation(control, bridge):
    cn, bn = control['nodes'], bridge['nodes']
    if not set(cn) <= set(bn) or any(bn[k] != value for k,value in cn.items()):
        raise ValueError('Existing nodes changed')
    ce = [(e['source_id'],e['target_id']) for e in control['edges']]
    be = [(e['source_id'],e['target_id']) for e in bridge['edges']]
    if be[:len(ce)] != ce or len(set(be)) != len(be):
        raise ValueError('Existing edges changed or duplicate edges')
    added = set(bn)-set(cn)
    if len(added)>1 or len(be)-len(ce)!=2*len(added):
        raise ValueError('Bridge cap or edge count violated')
    incoming, outgoing = Counter(), Counter()
    for s,t in be:
        if str(s) not in bn or str(t) not in bn or bn[str(t)]['t'] != bn[str(s)]['t']+1:
            raise ValueError('Invalid edge endpoints or time')
        outgoing[s]+=1; incoming[t]+=1
    if max(incoming.values(),default=0)>1 or max(outgoing.values(),default=0)>2:
        raise ValueError('Invalid lineage degree')
    for key in added:
        if incoming[int(key)]!=1 or outgoing[int(key)]!=1:
            raise ValueError('Added node is not a bridge')
        if any(not math.isfinite(bn[key][a]) or bn[key][a]<0 for a in ('t','z','y','x')):
            raise ValueError('Invalid added coordinates')
    return len(added)


def verify(root):
    path = root/'bridge_paired_result.json'
    result = json.loads(path.read_text())
    manifest = json.loads((root/'prelabel_graph_manifest.json').read_text())
    terminal = json.loads((root/'launcher_terminal.json').read_text())
    if result['run_id']!='focus-bridge-official-paired-v2' or result['status']!='completed':
        raise ValueError('Wrong or incomplete result')
    if terminal['status']!='completed' or terminal['run_id']!=result['run_id']:
        raise ValueError('Wrong or incomplete terminal')
    if result['authorized_for_submission'] is not False or result['metric_hack_used'] is not False:
        raise ValueError('Invalid authorization or metric policy')
    if set(manifest)!=STEMS or result['prepared']!=manifest:
        raise ValueError('Prelabel manifest mismatch')
    total_added = 0
    for stem in sorted(STEMS):
        graphs = {}
        for arm in ('control','bridge'):
            record = manifest[stem]['graphs'][arm]
            name = f'{stem}-{arm}.json'
            if Path(record['path']).name != name:
                raise ValueError('Unexpected graph filename')
            graph_path = root/'paired_graphs'/name
            if sha(graph_path)!=record['sha256']:
                raise ValueError('Graph hash mismatch')
            graphs[arm] = json.loads(graph_path.read_text())
        count = check_mutation(graphs['control'],graphs['bridge'])
        stats = manifest[stem]['stats']
        if stats['added_nodes']!=count or stats['added_edges']!=2*count:
            raise ValueError('Mutation counters disagree')
        total_added+=count
    calculated = {}
    for arm in ('control','bridge'):
        rows = result['per_movie'][arm]
        if len(rows)!=4 or {r['stem'] for r in rows}!=STEMS:
            raise ValueError('Movie coverage differs')
        for r in rows:
            if any(not math.isfinite(r[k]) or r[k]<0 or r[k]!=int(r[k])
                   for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')):
                raise ValueError('Invalid confusion counts')
            denom = r['edge_tp']+r['edge_fp']+r['edge_fn']
            if denom <= 0 or not math.isfinite(r['total_node_ratio']):
                raise ValueError('Missing per-movie metric inputs')
            edge = r['edge_tp']/denom
            adjusted = max(0.0, edge*(1-0.1*r['total_node_ratio']))
            if not same(edge,r['edge_jaccard']) or not same(adjusted,r['adj_edge_jaccard']):
                raise ValueError('Per-movie metric arithmetic differs')
        calculated[arm] = aggregate(rows)
        for key,value in calculated[arm].items():
            if not same(value,result['summaries'][arm][key]):
                raise ValueError('Aggregate arithmetic differs')
        for embryo in ('44b6','6bba'):
            subset = [r for r in rows if r['stem'].startswith(embryo+'_')]
            for key,value in aggregate(subset).items():
                if not same(value,result['by_embryo'][arm][embryo][key]):
                    raise ValueError('Per-embryo arithmetic differs')
    rows = {a: {r['stem']:r for r in result['per_movie'][a]} for a in calculated}
    delta = {s:rows['bridge'][s]['adj_edge_jaccard']-rows['control'][s]['adj_edge_jaccard'] for s in STEMS}
    gain = sum(rows['bridge'][s]['edge_tp']-rows['control'][s]['edge_tp'] for s in STEMS)
    if set(result['per_movie_adjusted_delta'])!=STEMS or any(
        not same(value,result['per_movie_adjusted_delta'][s]) for s,value in delta.items()
    ):
        raise ValueError('Reported movie deltas differ')
    bd, cd = calculated['bridge']['division_jaccard'],calculated['control']['division_jaccard']
    passed = bool(calculated['bridge']['score']>calculated['control']['score'] and min(delta.values())>=0
                  and gain>0 and (bd>=cd or (math.isnan(bd) and math.isnan(cd))))
    if result['paired_gate_passed'] is not passed or result['edge_tp_gain']!=gain:
        raise ValueError('Promotion calculation differs')
    return {'run_id':result['run_id'],'status':'verified_paired_pass' if passed else 'verified_rejection',
            'summaries':calculated,'added_bridges':total_added,'edge_tp_gain':gain,
            'result_sha256':sha(path),'authorized_for_submission':False,
            'authorized_for_production_promotion':False,
            'validation_scope':'paired diagnostic: all four movies occurred in public secondary model training',
            'requires_independent_generalization_evidence':True,
            'verification_scope':'graph hashes, mutations, reported confusion-count aggregation; not independent GT rescoring'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output_root',type=Path)
    args=parser.parse_args(); print(json.dumps(verify(args.output_root),indent=2))
