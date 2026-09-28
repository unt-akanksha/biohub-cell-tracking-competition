"""Independent audit of the frozen real-pair screening gate, not tracking score."""
import math


def checked_gate(before, after):
    for metrics in (before, after):
        if not math.isfinite(metrics['ce']) or metrics['ce'] < 0:
            raise ValueError('Invalid cross entropy')
        if metrics['total'] <= 0 or not 0 <= metrics['correct'] <= metrics['total']:
            raise ValueError('Invalid paired sample counts')
        for name, row in metrics['per_movie'].items():
            if not 0 <= row['correct'] <= row['total'] <= row['annotated']:
                raise ValueError('Inconsistent movie counts')
        if sum(r['correct'] for r in metrics['per_movie'].values()) != metrics['correct'] or sum(r['total'] for r in metrics['per_movie'].values()) != metrics['total']:
            raise ValueError('Pooled counts do not equal movie counts')
    if set(before['per_movie']) != set(after['per_movie']):
        raise ValueError('Movie inventory changed')
    if before['total'] != after['total']:
        raise ValueError('Eligibility changed')
    for name, row in before['per_movie'].items():
        other=after['per_movie'][name]
        if (row['total'],row['annotated']) != (other['total'],other['annotated']):
            raise ValueError('Per-movie denominator changed')
    return (after['correct'] > before['correct'] and after['ce'] < before['ce']
            and all(r['correct'] >= before['per_movie'][s]['correct'] for s,r in after['per_movie'].items()))
