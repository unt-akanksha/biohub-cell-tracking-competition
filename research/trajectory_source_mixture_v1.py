"""Fixed union of two source-trained ordinary-motion compatibility experts."""
import copy


def reconnect(final, coords, raw_edges, detector_ids, models, endpoint_function):
    if len(models) != 2:
        raise ValueError('Exactly two frozen source motion experts required')
    proposed = {}
    expert_counts = []
    for index, model in enumerate(models):
        graph, details = endpoint_function(final, coords, raw_edges, detector_ids,
                                            model, float(model['threshold']))
        if graph['nodes'] != final['nodes'] or details['added_nodes'] != 0:
            raise ValueError('Expert changed nodes')
        expert_counts.append(details['added_edges'])
        for record in details['links']:
            pair = (record['source_id'], record['target_id'])
            if pair not in proposed:
                proposed[pair] = dict(source_id=pair[0], target_id=pair[1], supporting_experts=[])
            proposed[pair]['supporting_experts'].append(index)
    records = [proposed[p] for p in sorted(proposed)]
    if len({r['source_id'] for r in records}) != len(records) or len({r['target_id'] for r in records}) != len(records):
        raise ValueError('Mixture creates conflicting endpoint assignments')
    result = copy.deepcopy(final)
    result['edges'].extend(dict(source_id=r['source_id'], target_id=r['target_id']) for r in records)
    return result, dict(added_nodes=0, added_edges=len(records), expert_added_edges=expert_counts,
                        links=records, movie_identifier_used=False, ground_truth_used=False)
