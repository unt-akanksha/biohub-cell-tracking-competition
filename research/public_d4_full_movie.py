"""Image-only execution adapters for the pinned public LF-DCTTA D4 ablation.

Never execute notebook cells, dataset searches, installers, proxy selection,
ground-truth readers, or public submission writers. Model/math definitions are
extracted verbatim by AST; the only predictor adaptation is its log directory.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

STEMS = ('44b6_24264f12', '44b6_81c256f0', '6bba_23af9eeb', '6bba_f1fde7e0')
NOTEBOOK_SHA = '95f08bb82388e9206f45c86daf926bb3d7a7fff9889c89b5fa9596f8e13c3bd4'
PREDICTOR_NAMES = ('PredictConfig', 'build_graph', '_load_frame',
                   'pool_kernel_from_um', '_detect_cells_pooled', 'predict_video')
POST_NAMES = ('edge_distance_um', 'point_distance_um', 'node_point', 'edge_sort_key',
              '_next_node_id', 'read_test_frame', 'refine_synthetic_midpoint',
              '_dc_pool_frame_xy', '_dc_normalize_dynamic_range', '_DCConvBlock3d',
              '_DCDeepCenterUNet3D', '_dc_cache_trim', 'deepcenter_heatmap_for_frame',
              'deepcenter_score_point', 'deepcenter_accept_repair_point', '_position_um',
              'motion_relink_edges', 'close_single_frame_gaps', '_single_successor_map',
              '_single_predecessor_map', 'recover_strict_gap2', 'add_safe_divisions_postlink',
              'filter_short_track_components', 'linefit_smooth_output_graph', 'filter_output_graph')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def public_config(notebook):
    """Resolve only primitive config assignments against a private environment."""
    if sha(notebook) != NOTEBOOK_SHA:
        raise ValueError('Pinned public source changed')
    cells = json.loads(Path(notebook).read_text(encoding='utf-8'))['cells']
    env = {}
    for index in (0, 3, 4):
        for node in ast.parse(''.join(cells[index]['source'])).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (isinstance(target, ast.Subscript) and ast.unparse(target.value) == 'os.environ'
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                key = ast.literal_eval(target.slice)
                if not key.startswith('BIOHUB_'):
                    raise ValueError('Unexpected environment key')
                env[key] = node.value.value
    prefixes = ('DET_', 'UNET_', 'USE_ILP', 'ILP_', 'OUTPUT_', 'MOTION_', 'DIV_',
                'GAP_', 'GAP2_', 'ADAPTIVE_', 'SHORT_', 'SAFE_', 'USE_DEEPCENTER',
                'REQUIRE_DEEPCENTER', 'DEEPCENTER_')
    namespace = {'os': SimpleNamespace(environ=env), '__builtins__': {'float': float, 'int': int}}
    keys = []
    for node in ast.parse(''.join(cells[2]['source'])).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith(prefixes)):
            # No arbitrary calls or names even if an upstream source is replaced.
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call) and ast.unparse(sub.func) not in ('float', 'int', 'os.environ.get'):
                    raise ValueError('Non-config call in config extraction')
                if isinstance(sub, ast.Name) and sub.id not in ('os', 'int', 'float'):
                    raise ValueError('Non-config name in config extraction')
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'literal-public-config', 'exec'), namespace)
            keys.append(node.targets[0].id)
    config = {key: namespace[key] for key in keys}
    if (len(config) != 82 or config['DET_THRESHOLD'] != .965
            or config['DEEPCENTER_SAFE_DIV_THRESHOLD'] != .25
            or env['BIOHUB_SECONDARY_LOW_MARGIN_MAX'] != '0.35'):
        raise ValueError('Frozen public configuration drift')
    return {'environment': env, 'globals': config}


def adapt_predictor_logs(source):
    # This is a path-only change. Do not suppress the public retention guard.
    needle = 'Path("/kaggle/working")'
    if source.count(needle) != 2:
        raise ValueError('Expected exactly two public predictor log directories')
    return source.replace(needle, '_RUN_OUTPUT')


def original_postprocess(corrected):
    a = 'torch_mod.rot90(tensor, 2, dims=(-2, -1))'
    b = 'torch_mod.rot90(model(at).transpose(-1, -2), -2, dims=(-2, -1))'
    if corrected.count(a) != 1 or corrected.count(b) != 1:
        raise ValueError('DeepCenter paired reversal changed')
    return corrected.replace(a, a.replace(', 2,', ', 1,'), 1).replace(b, b.replace(', -2, dims', ', -1, dims'), 1)


def image_metadata(path, *, normalize=False, load_image=False, downsample=(1, 4, 4)):
    if normalize or load_image or tuple(downsample) != (1, 4, 4):
        raise ValueError('Unexpected image-only loader request')
    path = Path(path)
    group = json.loads((path / 'zarr.json').read_text())['attributes']
    array = json.loads((path / '0/zarr.json').read_text())
    scale = group['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1:]
    if array['shape'] != [100, 64, 256, 256] or scale != [1.625, .40625, .40625]:
        raise ValueError('Unexpected full movie geometry')
    return SimpleNamespace(zarr_path=path, image_shape=(100, 64, 64, 64), scale=tuple(scale),
                           quantiles=group['image_statistics']['quantiles'])


def csv_equivalent_graph(nodes, edges, frames):
    """Match the public CSV's integer coordinates, then fail on invalid lineage."""
    result = {'nodes': {}, 'edges': []}
    for ident, node in nodes.items():
        ident = int(ident)
        if int(node['node_id']) != ident or not 0 <= int(node['t']) < frames:
            raise ValueError('Invalid node identity/time')
        row = {'node_id': ident, 't': int(node['t'])}
        for axis, bound in zip(('z', 'y', 'x'), (64, 256, 256)):
            value = max(0, int(round(float(node[axis]))))
            if value >= bound:
                raise ValueError('Out-of-volume predicted coordinate')
            row[axis] = value
        result['nodes'][str(ident)] = row
    parents, children, seen = Counter(), Counter(), set()
    for edge in edges:
        a, b = int(edge['source_id']), int(edge['target_id'])
        if ((a, b) in seen or str(a) not in result['nodes'] or str(b) not in result['nodes']
                or result['nodes'][str(b)]['t'] != result['nodes'][str(a)]['t'] + 1):
            raise ValueError('Duplicate, dangling or nonconsecutive edge')
        seen.add((a, b)); parents[b] += 1; children[a] += 1
        result['edges'].append({'source_id': a, 'target_id': b})
    if not nodes or not edges or max(parents.values()) > 1 or max(children.values()) > 2:
        raise ValueError('Empty or invalid biological lineage graph')
    if set(row['t'] for row in result['nodes'].values()) != set(range(frames)):
        raise ValueError('Predicted graph does not cover every requested frame')
    return result
