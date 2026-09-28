"""Prediction-only physical and neural evidence for joint lineage events."""
import numpy as np

from research.trajectory_candidate_ranker_v1 import FEATURES as EDGE_FEATURES
from research.trajectory_disagreement_data_v1 import positions, adjacency

FEATURES = EDGE_FEATURES + (
    'continuation_bias', 'division_bias', 'birth_bias', 'death_bias',
    'sister_distance', 'parent_distance_asymmetry', 'daughter_midpoint_motion_residual',
    'mother_history_known', 'daughter_future_separation_change', 'daughter_future_known',
    'birth_boundary_distance', 'death_boundary_distance')


def features(problem, graph, edge_matrix):
    edge_matrix = np.asarray(edge_matrix)
    if edge_matrix.ndim != 2 or edge_matrix.shape[1] != len(EDGE_FEATURES) or not np.isfinite(edge_matrix).all():
        raise ValueError('Invalid frozen edge feature schema')
    ix = problem['edge_indices']
    if ix.shape != (len(problem['options']), 2) or np.any(ix < -1) or np.any(ix >= len(edge_matrix)):
        raise ValueError('Invalid event edge lookup')
    result = np.zeros((len(ix), len(FEATURES)), np.float64)
    for slot in (0, 1):
        valid = ix[:,slot] >= 0
        result[valid, :len(EDGE_FEATURES)] += edge_matrix[ix[valid,slot]]
    nodes = {int(k): v for k, v in graph['nodes'].items()}
    pos = positions(nodes)
    incoming, outgoing = adjacency(graph['edges'])
    extent = np.array([63*1.625, 255*.40625, 255*.40625])
    def boundary(node):
        value = min(float(pos[node].min()), float((extent-pos[node]).min()))
        if value < -1e-6:
            raise ValueError('Node lies outside the observed image geometry')
        return min(20., max(0., value)) / 10.
    offset = len(EDGE_FEATURES)
    for index, (parent, first, second) in enumerate(problem['options']):
        if parent < 0:
            child = int(problem['children'][first])
            result[index, offset+2] = 1.
            result[index, offset+10] = boundary(child)
        elif first < 0:
            mother = int(problem['parents'][parent])
            result[index, offset+3] = 1.
            result[index, offset+11] = boundary(mother)
        elif second < 0:
            result[index, offset] = 1.
        else:
            mother = int(problem['parents'][parent])
            a, b = (int(problem['children'][c]) for c in (first, second))
            result[index, offset+1] = 1.
            separation = np.linalg.norm(pos[a]-pos[b])
            result[index, offset+4] = separation / 10.
            result[index, offset+5] = abs(np.linalg.norm(pos[a]-pos[mother])-np.linalg.norm(pos[b]-pos[mother])) / 10.
            if len(incoming[mother]) == 1:
                before = incoming[mother][0]
                if nodes[mother]['t']-nodes[before]['t'] == 1:
                    velocity = pos[mother]-pos[before]
                    result[index, offset+6] = np.linalg.norm((pos[a]+pos[b])/2-pos[mother]-velocity) / 10.
                    result[index, offset+7] = 1.
            if len(outgoing[a]) == len(outgoing[b]) == 1:
                after_a, after_b = outgoing[a][0], outgoing[b][0]
                if nodes[after_a]['t']-nodes[a]['t'] == nodes[after_b]['t']-nodes[b]['t'] == 1:
                    result[index, offset+8] = (np.linalg.norm(pos[after_a]-pos[after_b])-separation) / 10.
                    result[index, offset+9] = 1.
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite event evidence')
    return result.astype(np.float32)
