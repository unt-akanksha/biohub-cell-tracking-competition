from pathlib import Path
import runpy
import numpy as np
from research.image_context_inputs import anchor_tokens, contexts_for_patches
from research.image_division_context import image_context


def nodes():
    return {7:dict(t=30,z=20,y=40,x=60), 8:dict(t=31,z=21,y=36,x=56),
            9:dict(t=31,z=21,y=44,x=64)}


def test_anchor_layout_matches_frozen_source_exactly():
    original = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'research/temporal_contrastive/graph_context_features.py'))
    graph = nodes()
    context, _ = original['context_tokens'](original['physical_nodes'](graph),
        parent_id=7,existing_child_id=8,proposed_child_id=9)
    np.testing.assert_array_equal(anchor_tokens(graph,7,8,9),context[:3])
    np.testing.assert_array_equal(anchor_tokens(graph,7,9,8),context[:3])
    graph[999] = dict(t=-100,z=float('nan'),y=0,x=0)
    np.testing.assert_array_equal(anchor_tokens(graph,7,8,9),context[:3])


def test_float32_inference_replays_float16_training_preprocessing():
    rng = np.random.default_rng(19)
    patches = rng.normal(size=(2,3,3,17,17,17)).astype(np.float32)
    anchors = np.stack([anchor_tokens(nodes(),7,8,9)]*2)
    context, mask = contexts_for_patches(patches,anchors)
    for index in range(2):
        expected, expected_mask = image_context(patches[index,0].astype(np.float16),anchors[index])
        np.testing.assert_array_equal(context[index],expected.astype(np.float16))
        np.testing.assert_array_equal(mask[index],expected_mask)
