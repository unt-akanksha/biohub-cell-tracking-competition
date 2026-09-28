"""Independent release invariants for AR2 repair followed by structured swaps."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

from research.trajectory_runtime_v1 import sha, validate_graph

CODE_SHA = 'a07d605e09b375e973e4e56bf26faca4d1d4ce10390f40e5eb50d33077d5a661'
MODEL_SHA = 'c8411cf313a62f3a7d0f3550f68ca84d4dc97970cf31eefe3ff24bfe810402d4'


def pairs(graph):
    return {(int(e['source_id']), int(e['target_id'])) for e in graph['edges']}


def check_graph_stages(base, motion, final, frames, added_edges, details):
    for graph in (base, motion, final):
        validate_graph(graph, frames)
    if base['nodes'] != motion['nodes'] or motion['nodes'] != final['nodes']:
        raise ValueError('Cell coordinates or node identities changed')
    old, middle, new = map(pairs, (base, motion, final))
    if not old <= middle or len(middle - old) != added_edges:
        raise ValueError('Endpoint stage is not the recorded add-only repair')
    for axis in (0, 1):
        if Counter(e[axis] for e in middle) != Counter(e[axis] for e in new):
            raise ValueError('Structured stage changed lineage degrees')
    outgoing = Counter(a for a, b in middle)
    nodes = motion['nodes']
    protected = set()
    for a, b in middle:
        if outgoing[a] == 2 or nodes[str(b)]['t'] - nodes[str(a)]['t'] != 1:
            protected.update((a, b))
    def incident(edges):
        return {e for e in edges if e[0] in protected or e[1] in protected}
    if incident(middle) != incident(new):
        raise ValueError('Division or gap incident links changed')
    changed = len(new - middle)
    if changed != details['changed_edges']:
        raise ValueError('Structured edit count differs from log')
    for flag in ('node_positions_unchanged', 'degrees_unchanged',
                 'division_and_gap_incident_edges_unchanged'):
        if details[flag] is not True:
            raise ValueError('Reported preservation check failed: ' + flag)
    return changed


def load_portable(folder):
    folder = Path(folder)
    code = folder / 'structured-trajectory.py'
    model_path = folder / 'structured-trajectory-model.json'
    if sha(code) != CODE_SHA or sha(model_path) != MODEL_SHA:
        raise ValueError('Frozen portable model/code changed')
    spec = importlib.util.spec_from_file_location('structured_release_portable', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = json.loads(model_path.read_text(encoding='utf-8'))
    if tuple(model['features']) != module.FEATURES:
        raise ValueError('Feature schema changed')
    return module, model['weights']


def replay_movie(folder, module, weights, final):
    import numpy as np
    folder = Path(folder)
    initial = json.loads((folder / 'pre-postprocess.json').read_text())
    motion = json.loads((folder / 'motion-repaired-prediction.json').read_text())
    with np.load(folder / 'raw-candidates.npz', allow_pickle=False) as raw:
        replay, details = module.refine(initial, motion, raw['coords'], raw['edges'], weights)
    if replay != final:
        raise ValueError('Independent frozen-model replay differs from production graph')
    return details


def runtime_evidence(prior_times, public_times, validation_projection, public_wall):
    if len(prior_times) != 8 or len(public_times) != 4:
        raise ValueError('Complete eight-validation/four-public timing evidence required')
    if not all(math.isfinite(v) and v > 0 for v in [*prior_times, *public_times, validation_projection, public_wall]):
        raise ValueError('Invalid timing evidence')
    # Calibrate the already-measured four-worker/two-GPU schedule using its
    # complete makespan, including contention, startup, and assembly. Dividing
    # overlapping worker wall times by two falsely double-counts concurrency;
    # dividing by four without overhead is optimistic. Neither is used here.
    validation_wall = validation_projection * 8 * 3600 / 199
    effective_workers = sum(prior_times) / validation_wall
    if not 0 < effective_workers <= 4:
        raise ValueError('Timing evidence exceeds the actual four-worker schedule')
    public = sum(public_times) / 4 * 199 / effective_workers / 3600
    combined = (validation_wall + public_wall) / 12 * 199 / 3600
    if validation_projection > 8 or max(public, combined) >= 10 or public_wall >= 36000:
        raise ValueError('Observed timing fails the frozen ten-hour runtime policy')
    return dict(validation_cohort_projection_hours=validation_projection,
                validation_calibrated_public_work_projection_hours=public,
                combined_twelve_movie_makespan_projection_hours=combined,
                public_four_movie_makespan_projection_hours=public_wall / 4 * 199 / 3600,
                effective_workers_measured_on_two_gpus=effective_workers,
                runtime_policy_revision=3,
                assumes_validation_scheduling_efficiency_transfers=True,
                worker_times_include_overlap_and_waiting=True,
                hidden_inference_cutoff_hours=10,
                hidden_movie_count_projection_assumption=199,
                dense_movie_and_static_shard_imbalance_timeout_risk=True,
                hidden_runtime_guaranteed=False)
