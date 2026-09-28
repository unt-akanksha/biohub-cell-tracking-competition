"""Exactly two GPUs; at most two independent whole-movie workers on each."""
from submission_sharding import MovieShard,build_movie_shards


def build_movie_slots(movie_ids,cuda_tokens,*,movie_weights):
    parents=build_movie_shards(movie_ids,cuda_tokens,movie_weights=movie_weights)
    slots=[]
    for parent in parents:
        if len(parent.movie_ids)<2:
            children=(parent.movie_ids,)
        else:
            local={m:movie_weights[m] for m in parent.movie_ids}
            split=build_movie_shards(parent.movie_ids,('slot0','slot1'),movie_weights=local)
            children=tuple(s.movie_ids for s in split)
        for movies in children:
            slots.append((parent.cuda_token,movies))
    return tuple(MovieShard(i,len(slots),gpu,tuple(movies)) for i,(gpu,movies) in enumerate(slots))


def validate_slot_outputs(slots,observed):
    indices=set(range(len(slots)))
    if (len(slots) not in (2,3,4) or {s.shard_index for s in slots}!=indices
            or set(observed)!=indices or any(s.shard_count!=len(slots) or not s.movie_ids for s in slots)):
        raise ValueError('Missing or invalid work slots')
    tokens={s.cuda_token for s in slots}
    if len(tokens)!=2 or any(sum(s.cuda_token==t for s in slots)>2 for t in tokens):
        raise ValueError('Exactly two GPUs and at most two slots each required')
    expected=[m for s in slots for m in s.movie_ids]
    if len(expected)!=len(set(expected)):
        raise ValueError('Duplicate planned movie')
    for slot in slots:
        rows=observed[slot.shard_index]
        if len(rows)!=len(set(rows)) or set(rows)!=set(slot.movie_ids):
            raise ValueError('Duplicated, missing or extra movie output')
    return tuple(sorted(expected))
