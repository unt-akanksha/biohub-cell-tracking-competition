from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from research.trajectory_structured_release_v1 import (
    check_graph_stages, load_portable, replay_movie, runtime_evidence)

ROOT = Path(__file__).resolve().parents[1]


def graph():
    return dict(nodes={str(i): dict(node_id=i, t=int(i >= 2), z=0, y=0, x=i) for i in range(4)},
                edges=[dict(source_id=0, target_id=2), dict(source_id=1, target_id=3)])


def details(changed=2):
    return dict(changed_edges=changed, node_positions_unchanged=True,
                degrees_unchanged=True, division_and_gap_incident_edges_unchanged=True)


def swapped():
    result = graph()
    result['edges'] = [dict(source_id=0, target_id=3), dict(source_id=1, target_id=2)]
    return result


def test_legitimate_swaps_are_not_rejected_as_non_additive():
    assert check_graph_stages(graph(), graph(), swapped(), 2, 0, details()) == 2


def test_node_tampering_rejected():
    result = swapped()
    result['nodes']['0']['x'] += 1
    with pytest.raises(ValueError, match='coordinates'):
        check_graph_stages(graph(), graph(), result, 2, 0, details())


def test_new_division_rejected_even_when_graph_is_valid():
    result = swapped()
    result['edges'][1]['source_id'] = 0
    with pytest.raises(ValueError, match='degrees'):
        check_graph_stages(graph(), graph(), result, 2, 0, details())


def test_division_incident_swap_rejected_despite_same_degrees():
    original = graph()
    original['nodes']['4'] = dict(node_id=4, t=1, z=0, y=0, x=4)
    original['edges'].append(dict(source_id=0, target_id=4))
    result = deepcopy(original)
    result['edges'][0]['source_id'] = 1
    result['edges'][1]['source_id'] = 0
    with pytest.raises(ValueError, match='Division'):
        check_graph_stages(original, original, result, 2, 0, details())


def test_add_only_stage_and_edit_counts_are_verified():
    with pytest.raises(ValueError, match='add-only'):
        check_graph_stages(graph(), swapped(), swapped(), 2, 0, details(0))
    with pytest.raises(ValueError, match='edit count'):
        check_graph_stages(graph(), graph(), swapped(), 2, 0, details(1))


def test_runtime_policy_does_not_count_four_workers_as_four_gpus():
    evidence = runtime_evidence([200.] * 8, [300.] * 4, 6., 400.)
    assert evidence['validation_calibrated_public_work_projection_hours'] == pytest.approx(9.)
    assert not evidence['hidden_runtime_guaranteed']
    with pytest.raises(ValueError, match='runtime policy'):
        runtime_evidence([200.] * 8, [400.] * 4, 6., 600.)
    with pytest.raises(ValueError, match='Invalid timing'):
        runtime_evidence([float('nan')] * 8, [300.] * 4, 6.6, 400.)


def test_production_only_changes_mode_and_remote_code_is_exact():
    spec = importlib.util.spec_from_file_location('structured_submit', ROOT / 'scripts/submit-trajectory-structured-v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = json.loads((ROOT / 'reports/experiments/trajectory-structured-production-v1-build.json').read_text())
    stage = ROOT / '.biohub/staging/biohub-structured-trajectory-candidate-v1'
    module.verify_source(stage, ROOT / '.biohub/cache/trajectory-structured-production-v1-source', build)
    prior = ROOT / '.biohub/staging/biohub-structured-trajectory-acceptance-v1/structured-trajectory-acceptance.ipynb'
    accepted = ''.join(json.loads(prior.read_text())['cells'][1]['source'])
    produced = ''.join(json.loads((stage / 'structured-trajectory-candidate.ipynb').read_text())['cells'][1]['source'])
    assert accepted.count("RUN_MODE = 'acceptance'") == 1
    assert produced == accepted.replace("RUN_MODE = 'acceptance'", "RUN_MODE = 'production'", 1)


def test_full_eight_t4_graphs_independently_replay():
    folder = ROOT / '.biohub/cache/trajectory-structured-kaggle-v1-output/trajectory-complete'
    terminal = json.loads((folder / 'result.json').read_text())
    portable, weights = load_portable(ROOT / '.biohub/cache/trajectory-structured-portable-v1')
    count = 0
    for shard, worker in terminal['workers'].items():
        for movie, arms in worker['movies'].items():
            path = folder / ('shard-' + shard) / (movie + '-original')
            read = lambda name: json.loads((path / name).read_text())
            final = read('repaired-prediction.json')
            changes = check_graph_stages(read('prediction.json'), read('motion-repaired-prediction.json'),
                                        final, 100, arms['original']['added_edges'],
                                        read('repair-details.json')['structured_assignment'])
            assert replay_movie(path, portable, weights, final)['changed_edges'] == changes
            count += 1
    assert count == 8
