"""Install hash-checked raw FOCUS detections into an official predictor.

Intended for one sequential movie per worker process. No labels, density
estimates, public predictions, threshold tuning, or node pruning are used.
"""
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

try:
    from focus_linker_cache import FocusLinkerCache
except ModuleNotFoundError:
    from research.focus_linker_cache import FocusLinkerCache


def load_raw_cache(root, stem, expected_sha256, actual_shape):
    root = Path(root)
    if not stem or Path(stem).name != stem or '/' in stem or '\\' in stem:
        raise ValueError('invalid movie name')
    path = root / f'{stem}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('raw detections hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'coords', 'movie_shape', 'scale_um'}:
            raise ValueError('unexpected raw checkpoint schema')
        shape = tuple(data['movie_shape'].tolist())
        if shape != tuple(actual_shape):
            raise ValueError('raw detections/image shape mismatch')
        coords = np.asarray(data['coords'])
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError('invalid coordinate array')
        # Do not sort here: checkpoint row identity must remain stable for edges.
        return FocusLinkerCache(tuple(range(len(coords))), coords, shape,
                                coordinate_source='raw instance centroids')


def solve_links_keep_nodes(graph, solver):
    """Select ILP edges without accepting its node-selection subgraph."""
    import tracksdata as td
    import polars as pl
    before = graph.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x']).sort('node_id')
    solver.solve(graph)
    selected = graph.edge_attrs().filter(pl.col(solver.output_key))
    # tracksdata's edge-filter subgraph also drops isolated nodes. Rebuild
    # explicitly from every input node and only the solver-selected edges.
    result = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        result.add_node_attr_key(axis, pl.Float64, 0.)
    new_ids = result.bulk_add_nodes(before.drop('node_id').to_dicts())
    mapping = dict(zip(before['node_id'].to_list(), new_ids))
    edge_keys = [key for key in ('edge_prob', 'edge_dist') if key in selected.columns]
    for key in edge_keys:
        result.add_edge_attr_key(key, pl.Float64, 0.)
    if len(selected):
        result.bulk_add_edges([
            dict(source_id=mapping[row['source_id']], target_id=mapping[row['target_id']],
                 **{key: row[key] for key in edge_keys})
            for row in selected.iter_rows(named=True)])
    after = result.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x']).sort('node_id')
    if not before.drop('node_id').equals(after.drop('node_id')):
        raise ValueError('ILP edge-only filtering changed raw detections')
    return result


def install(namespace, raw_root, manifest, shape_reader):
    """Wrap predict_video; validate every checkpoint before image inference.

    ``manifest`` must be a hash-bound terminal loaded by the launcher.
    ``shape_reader`` only reads the image Zarr shape, never GEFF or labels.
    """
    if manifest.get('status') != 'completed' or manifest.get('ground_truth_opened') is not False:
        raise ValueError('raw detector terminal is not complete and label-blind')
    records = manifest['raw_detections']
    hashes = {r['stem']: r['sha256'] for r in records}
    if len(hashes) != len(records) or not records:
        raise ValueError('invalid movie coverage')
    if any(r['failed_frames'] != 0 or r['postprocessing_applied'] is not False for r in records):
        raise ValueError('incomplete or processed detections')
    original = namespace['predict_video']
    detector = namespace['_detect_cells_pooled']
    if getattr(original, '_focus_raw_installed', False):
        raise ValueError('raw detector adapter already installed')
    signature = inspect.signature(original)

    @wraps(original)
    def predict(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        path = Path(bound.arguments['ds_path'])
        stem = path.stem
        if stem not in hashes:
            raise ValueError('movie outside frozen raw-detection manifest')
        if bound.arguments.get('max_frames') is not None:
            raise ValueError('complete movies required')
        if tuple(bound.arguments.get('downsample', (1, 4, 4))) != (1, 4, 4):
            raise ValueError('association grid changed')
        # The official CLI passes bare stems; its loader adds .zarr internally.
        image_path = path if path.suffix == '.zarr' else path.with_name(path.name + '.zarr')
        cache = load_raw_cache(raw_root, stem, hashes[stem], shape_reader(image_path))
        def detect(_logits, frame, _threshold, _pool_kernel):
            return cache.association_coords(frame, mode='rounded')
        namespace['_detect_cells_pooled'] = detect
        try:
            coords, edges = original(*args, **kwargs)
            restored = cache.restore_precise_output(coords)
            for edge in edges:
                source, target = edge[:2]
                if (int(source) != source or int(target) != target or
                        not 0 <= source < len(restored) or not 0 <= target < len(restored)):
                    raise ValueError('linker returned invalid edge identity')
                if restored[int(target), 0] != restored[int(source), 0] + 1:
                    raise ValueError('linker returned nonconsecutive edge')
            return restored, edges
        finally:
            namespace['_detect_cells_pooled'] = detector
    predict._focus_raw_installed = True
    namespace['predict_video'] = predict
    return predict
