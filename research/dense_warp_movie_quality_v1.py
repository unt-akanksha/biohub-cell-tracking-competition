"""Frozen full-movie diagnostic comparisons; invalid/incomplete data fail closed."""
import math


def compare(summaries, embryos, movies):
    arms={'original','warp44','warp6'}
    if any(set(value)!=arms for value in (summaries,embryos,movies)):
        raise ValueError('All three fixed arms required')
    stems=set(movies['original'])
    if len(stems)!=4 or any(set(movies[a])!=stems for a in arms):
        raise ValueError('Four paired complete movie summaries required')
    for arm in arms:
        if set(embryos[arm])!={'44b6','6bba'} or summaries[arm]['n']!=4 or summaries[arm]['n_adj']!=4:
            raise ValueError('Missing embryo or silently skipped metric row')
        for row in (summaries[arm],*embryos[arm].values(),*movies[arm].values()):
            if not all(math.isfinite(row[k]) for k in ('score','edge_jaccard','adj_edge_jaccard')):
                raise ValueError('Nonfinite score or edge metric')
    result={}
    for arm in ('warp44','warp6'):
        delta={k:summaries[arm][k]-summaries['original'][k] for k in ('score','edge_jaccard','adj_edge_jaccard')}
        per_embryo={e:embryos[arm][e]['score']-embryos['original'][e]['score'] for e in ('44b6','6bba')}
        per_movie={s:movies[arm][s]['score']-movies['original'][s]['score'] for s in sorted(stems)}
        gates=dict(pooled_score_strict=delta['score']>0,raw_edges_strict=delta['edge_jaccard']>0,
                   embryo_nonregression=min(per_embryo.values())>=0,movie_nonregression=min(per_movie.values())>=0)
        result[arm]=dict(delta=delta,embryo_delta=per_embryo,movie_delta=per_movie,gates=gates,diagnostic_pass=all(gates.values()))
    return result
