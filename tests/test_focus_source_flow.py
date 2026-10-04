import ast
from copy import deepcopy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
G = runpy.run_path(str(ROOT / 'research/focus_source_flow_contract.py'))
B = runpy.run_path(str(ROOT / 'scripts/build-focus-source-flow.py'))
C = runpy.run_path(str(ROOT / 'research/focus_source_flow_comparison.py'))


def inputs():
    split = (ROOT / 'research/independent_real_baseline_v1_split.json').read_bytes()
    stems = json.loads(split)['folds'][0]['selection']
    records = [dict(stem=s, frames=3, scope='training_smoke_replay', nodes=1, sha256='0'*64)
               for s in ['6bba_f1fde7e0', G['PROBE_STEM']]]
    records += [dict(stem=s, frames=100, scope='source_selection', nodes=1, sha256='0'*64) for s in stems]
    cache = dict(status='verified_raw_focus_source_cache', run_id='focus-source-cache-v1',
        notebook_sha256=G['CACHE_NOTEBOOK_SHA'], model_sha256=G['MODEL_SHA'], split_sha256=G['SPLIT_SHA'],
        source_stems=stems, complete_source_frames=800, smoke_frames_replayed=6,
        ground_truth_opened=False, authorized_for_submission=False,
        terminal=dict(status='completed', exact_smoke_replay=True), terminal_sha256='0'*64, records=records)
    return json.dumps(cache).encode(), split


def test_contract_training_replay_first_and_no_target():
    policy = G['receipt'](*inputs())
    assert policy['inference_stems'] == [G['PROBE_STEM']] + policy['source_stems']
    assert len(policy['source_stems']) == 8
    assert policy['flow_views'] == 1 and not policy['public_linker_loaded']
    assert policy['new_target_movies_opened'] == 0 and not policy['authorized_for_submission']


@pytest.mark.parametrize('field,value', [('complete_source_frames', 799), ('ground_truth_opened', True),
                                       ('model_sha256', 'wrong'), ('status', 'running')])
def test_contract_rejects_invalid_cache(field, value):
    cache, split = inputs()
    data = json.loads(cache)
    data[field] = value
    with pytest.raises(ValueError, match='cache'):
        G['receipt'](json.dumps(data).encode(), split)


def test_builder_reuses_exact_tested_flow_components():
    nb, meta = B['build'](inputs()[0])
    def runtime(nb):
        node = next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_sources')
        return ast.literal_eval(node.value)
    old = json.loads((ROOT / 'kaggle/biohub-focus-owned-flow-full-v1/biohub-focus-owned-flow-full-v1.ipynb').read_text())
    current, original = runtime(nb), runtime(old)
    for key in ('backward_flow_model.py', 'backward_flow_ops.py', 'raw_centroid_flow_sampling.py',
                'backward_flow_linking.py', 'independent_motion_prior.py', 'probe_runtime.py'):
        assert current[key] == original[key]
    assert 'range(0, frames - 1, 2)' in current['run_pilot.py']
    assert "if not is_probe and not probe_replayed" in current['run_pilot.py']
    assert 'require_tracks=False' in current['run_pilot.py']
    assert "np.array_equal(flows, probe_flow)" in current['run_pilot.py']
    assert meta['kernel_sources'][1] == 'indarkarhana/biohub-focus-source-cache-v1/1'
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert 'focus_source_flow' in ''.join(nb['cells'][0]['source'])


def comparisons():
    stems = G['receipt'](*inputs())['source_stems']
    parent = [dict(stem=s, adj_edge_jaccard=.6, num_pred_nodes=100, node_recall=.95) for s in stems]
    rows = dict(parent=deepcopy(parent), control=deepcopy(parent), candidate=deepcopy(parent))
    for r in rows['candidate']:
        r['adj_edge_jaccard'] = .61
    summaries = {a: dict(score=.6, edge_jaccard=.6, node_recall=.95) for a in rows}
    summaries['candidate'].update(score=.61, edge_jaccard=.61)
    return rows, summaries, stems


def test_declared_gate_pass_is_not_submission_authority():
    result = C['compare'](*comparisons())
    assert result['source_gate_passed'] and not result['authorized_for_submission']


def test_fewer_than_five_improving_movies_rejected():
    rows, summaries, stems = comparisons()
    for r in rows['candidate'][:4]:
        r['adj_edge_jaccard'] = .6
    assert not C['compare'](rows, summaries, stems)['source_gate_passed']


def test_recall_loss_and_static_non_gain_rejected():
    rows, summaries, stems = comparisons()
    summaries['candidate']['node_recall'] = .94
    assert not C['compare'](rows, summaries, stems)['source_gate_passed']
    rows, summaries, stems = comparisons()
    summaries['control']['score'] = .61
    assert not C['compare'](rows, summaries, stems)['source_gate_passed']


def test_per_movie_guard_and_raw_node_change_rejected():
    rows, summaries, stems = comparisons()
    rows['candidate'][0]['adj_edge_jaccard'] = .57
    assert not C['compare'](rows, summaries, stems)['source_gate_passed']
    rows['candidate'][0]['num_pred_nodes'] = 99
    with pytest.raises(ValueError, match='identical'):
        C['compare'](rows, summaries, stems)
