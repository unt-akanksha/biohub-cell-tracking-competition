"""Replay all cached graphs before paired current-official source scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_source_flow_contract import receipt, verify_raw_cache
from research.focus_source_flow_comparison import compare
from research.independent_motion_prior import link_motion
from research.backward_flow_linking import link_backward_flow

RUN = 'focus-source-flow-v1'
SCORER = runpy.run_path(str(ROOT / 'scripts/score-independent-selection.py'))
FULL = runpy.run_path(str(ROOT / 'scripts/score-focus-owned-flow-full.py'))
WORKER = runpy.run_path(str(ROOT / 'scripts/run-focus-source-flow.py'))
PARENT_MANIFEST_SHA = 'f77b9eff4b965afcd9e4f571e4e32f09f23c894e962bb712ed66b83e75b35679'
PARENT_REPORT_SHA = 'db9d75ad43a9bc74d3f38f3aef48e5e617510a6abb93205a0387e726415d080f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_graph(graph, coords, edges):
    import numpy as np
    import tracksdata as td
    nodes = graph.node_attrs().sort('node_id')
    if not np.array_equal(nodes.select('t', 'z', 'y', 'x').to_numpy(), coords):
        raise ValueError('Serialized graph changed raw centroid identity or order')
    mapping = {v: i for i, v in enumerate(nodes['node_id'])}
    columns = [td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
    actual = [] if graph.num_edges() == 0 else sorted((mapping[s], mapping[t])
        for s, t in graph.edge_attrs(attr_keys=columns).select(columns).iter_rows())
    if actual != sorted((s, t) for s, t, probability in edges):
        raise ValueError('Serialized graph differs from exact frozen linking replay')


def verify_sampling(record, coords, flows, frames, probe_stem):
    import numpy as np
    if (flows.shape != (len(coords), 3) or not np.isfinite(flows).all()
            or np.any(flows[coords[:, 0] == 0] != 0)
            or record['processed_frames'] != frames or record['processed_pairs'] != frames-1
            or record['image_shape'] != [frames,64,256,256] or record['all_nodes_covered'] is not True
            or record['coordinate_sha256'] != hashlib.sha256(coords.tobytes()).hexdigest()
            or len(record['sampler_receipts']) != frames-1
            or record['probe_replayed'] is not (record['stem'] == probe_stem)):
        raise ValueError('Complete finite exact-coordinate sampled motion required')
    for t, sample in enumerate(record['sampler_receipts'], 1):
        points = coords[coords[:, 0] == t, 1:]
        if (sample['frame'] != t or sample['sampled_nodes'] != len(points)
                or sample['trailing_border_extended_nodes'] != int(np.any(points/[1,4,4] > [63,63,63], axis=1).sum())
                or sample['policy'] != 'constant flow extension through trailing unsampled voxel centers'
                or sample['coordinates_modified'] is not False or sample['nodes_deleted'] is not False):
            raise ValueError('Boundary sampling or node-coverage receipt changed')


def prepare(folder, notebook, notebook_sha):
    import numpy as np
    import tracksdata as td
    if sha(notebook) != notebook_sha:
        raise ValueError('Launch-pinned source-flow notebook changed')
    nb = json.loads(notebook.read_text())
    bundles = SCORER['PILOT']['embedded_sources'](notebook)
    for name, directory, filename in [('sources', 'repo', 'source_hashes.json'),
                                       ('runtime_sources', 'runtime', 'runtime_hashes.json')]:
        expected = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in bundles[name].items()}
        if json.loads((folder / filename).read_text()) != expected:
            raise ValueError('Executed source manifest differs from frozen notebook')
        for path, digest in expected.items():
            if sha(folder / directory / path) != digest:
                raise ValueError('Executed source bytes changed')
    # Local replay must use the same two algorithms executed by the GPU worker.
    for name in ('independent_motion_prior', 'backward_flow_linking'):
        local = (ROOT / 'research' / (name + '.py')).read_text().replace('from research.independent_motion_prior import', 'from independent_motion_prior import')
        if local != bundles['runtime_sources'][name + '.py']:
            raise ValueError('Host linking implementation differs from executed frozen source')
    policy = receipt((folder / 'runtime/verified_source_cache.json').read_bytes(), (folder / 'runtime/split.json').read_bytes())
    if nb['metadata']['codex']['contract'] != policy:
        raise ValueError('Source contract differs from frozen launch')
    raw_root = ROOT / '.biohub/cache/kernel-outputs/focus-source-cache-v1'
    raw = verify_raw_cache(raw_root, policy)
    terminal = json.loads((folder / 'launcher_terminal.json').read_text())
    manifest = json.loads((folder / 'outputs/flow_manifest.json').read_text())
    if (terminal['status'] != 'completed' or terminal['run_id'] != RUN
            or terminal['declared_budget_seconds'] != 3600 or not 0 < terminal['elapsed_seconds'] <= 3600
            or terminal['submission_performed'] is not False
            or manifest['status'] != 'completed' or manifest['run_id'] != RUN or manifest['contract'] != policy
            or manifest['ground_truth_opened'] is not False or manifest['target_embryo_images_read'] is not False
            or manifest['new_target_movies_opened'] != 0 or manifest['authorized_for_submission'] is not False
            or manifest['probe_replayed'] is not True
            or manifest['frozen_flow_before'] != policy['flow_tensor_sha256']
            or manifest['frozen_flow_after'] != policy['flow_tensor_sha256']
            or [r['stem'] for r in manifest['records']] != policy['inference_stems']):
        raise ValueError('Complete bounded source-only immutable-flow execution required')
    probe_path = ROOT / '.biohub/cache/kernel-outputs/focus-owned-flow-probe-v1/focus_owned_flow_probe/outputs/sampled_flow.npz'
    if sha(probe_path) != policy['probe_motion_sha256']:
        raise ValueError('Original successful motion smoke changed')
    with np.load(probe_path, allow_pickle=False) as data:
        probe_coords, probe_flow = data['coords'].copy(), data['backward_um'].copy()
    prepared = {}
    for record in manifest['records']:
        stem = record['stem']
        frames = 3 if stem == policy['probe_stem'] else 100
        if record['raw_checkpoint_sha256'] != raw[stem]['sha256']:
            raise ValueError('Incorrect source centroid identity')
        with np.load(raw_root / 'raw_detections' / (stem + '.npz'), allow_pickle=False) as data:
            coords = data['coords'].copy()
        movie = folder / 'outputs' / stem
        sample = movie / 'sampled_flow.npz'
        if sha(sample) != record['sample_sha256']:
            raise ValueError('Sampled flow bytes changed')
        with np.load(sample, allow_pickle=False) as data:
            if set(data.files) != {'coords', 'backward_um'} or not np.array_equal(data['coords'], coords):
                raise ValueError('Sampled motion changed centroid identity/order')
            flows = data['backward_um'].copy()
        verify_sampling(record, coords, flows, frames, policy['probe_stem'])
        if stem == policy['probe_stem'] and (not np.array_equal(coords, probe_coords) or not np.array_equal(flows, probe_flow)):
            raise ValueError('Actual source-route GPU probe replay differs')
        expected_edges = dict(control=link_motion(coords), candidate=link_backward_flow(coords, flows))
        arms = {}
        for arm, edges in expected_edges.items():
            path = movie / (arm + '.geff')
            if WORKER['tree_hash'](path) != record['graphs'][arm]['graph_sha256']:
                raise ValueError('Graph artifact checksum mismatch')
            graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
            verify_graph(graph, coords, edges)
            if record['graphs'][arm]['nodes'] != len(coords) or record['graphs'][arm]['edges'] != len(edges):
                raise ValueError('Graph count receipt differs from actual output')
            arms[arm] = path
        prepared[stem] = arms
    parent_root = ROOT / '.biohub/cache/kernel-outputs/detector-spatial-tta-selection-v1/detector_spatial_tta_selection'
    if sha(parent_root / 'outputs/selection_manifest.json') != PARENT_MANIFEST_SHA:
        raise ValueError('Frozen parent prediction manifest changed')
    parent_graphs, _ = SCORER['prepare'](parent_root,
        ROOT / 'kaggle/biohub-detector-spatial-tta-selection-v1/biohub-detector-spatial-tta-selection-v1.ipynb')
    if list(parent_graphs) != policy['source_stems']:
        raise ValueError('Parent graph source scope mismatch')
    return prepared, parent_graphs, policy


def score(folder, notebook, notebook_sha):
    import tracksdata as td
    from geff import GeffMetadata
    prepared, parent_graphs, policy = prepare(folder, notebook, notebook_sha)
    parent_path = ROOT / 'reports/experiments/detector-spatial-tta-selection-v1-score.json'
    if sha(parent_path) != PARENT_REPORT_SHA:
        raise ValueError('Retained source comparison changed')
    parent_report = json.loads(parent_path.read_text())['result']
    metric = SCORER['load_scorer'](ROOT / '.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    rows = {a: [] for a in ('parent', 'control', 'candidate')}
    for stem in policy['source_stems']:
        gt_path = ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train' / (stem + '.geff')
        for arm in rows:
            graph = parent_graphs[stem] if arm == 'parent' else td.graph.IndexedRXGraph.from_geff(str(prepared[stem][arm]))[0]
            truth = td.graph.IndexedRXGraph.from_geff(str(gt_path))[0]
            er = metric.evaluate(graph, truth, scale=(1.625,.40625,.40625), max_distance=7.)
            count = float(GeffMetadata.read(str(gt_path)).extra['estimated_number_of_nodes'])
            row = dict(metric.per_sample_metrics(er, count, SCORER['diagnostic_node_recall'](metric, graph, truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(dict(arm=arm, **row)), flush=True)
    # The old control must reproduce on the current GT and exact official scorer.
    if rows['parent'] != parent_report['per_movie']:
        raise ValueError('Fresh parent rescore differs from the original fixed source reference')
    summaries = {a: metric.summarise(v) for a, v in rows.items()}
    if summaries['parent'] != parent_report['summary']:
        raise ValueError('Fresh parent aggregate differs from original reference')
    return dict(status='completed_focus_source_flow_comparison', run_id=RUN, contract=policy,
        per_movie=rows, summaries=summaries,
        by_embryo={a: {e: metric.summarise([r for r in v if r['embryo'] == e]) for e in sorted({r['embryo'] for r in v})} for a, v in rows.items()},
        comparison=compare(rows, summaries, policy['source_stems']),
        graph_and_motion_cpu_replay=True, parent_rescore_exact=True,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        notebook_sha256=notebook_sha, manifest_sha256=sha(folder / 'outputs/flow_manifest.json'),
        new_target_movies_opened=0, authorized_for_submission=False,
        caveat='Complete fixed source-selection movies excluded from owned-flow training; FOCUS pretraining overlap unverified. Not leaderboard evidence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook-sha256', required=True)
    args = parser.parse_args()
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    if target.exists():
        raise ValueError('Refuse to overwrite completed source comparison')
    result = score(ROOT / f'.biohub/cache/kernel-outputs/{RUN}/focus_source_flow',
                   ROOT / f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb', args.notebook_sha256)
    target.write_text(json.dumps(FULL['finite_json'](result), indent=2, allow_nan=False))
    print(json.dumps(result['comparison'], indent=2))
