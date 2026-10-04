"""Validate raw-node preservation, then score complete graphs on CPU.

The current organizer scorer is pinned by the replay helper. All candidate
and control predictions are verified before any ground-truth graph is opened.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus3d_bridge_rescue import _validate_graph
REPLAY = runpy.run_path(str(ROOT / 'scripts/replay-focus-bridge-official.py'))
RAW = runpy.run_path(str(ROOT / 'scripts/verify-focus-raw-detections.py'))


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b'\0'); digest.update(path.read_bytes()); digest.update(b'\0')
    return digest.hexdigest()


def canonical(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 4)
    return points[np.lexsort(tuple(points[:, i] for i in (3, 2, 1, 0)))]


def edge_support(node_matches, predicted_edges, truth_edges):
    """Diagnostic only: partition annotated links using organizer node matches.

    This never edits predictions or provides oracle-selected ensemble edges.
    Unannotated detections cannot be classified as false positives here.
    """
    matches = {int(k): int(v) for k, v in node_matches.items()
               if v is not None and int(v) >= 0}
    truth_edges = set(truth_edges)
    represented = set(matches.values())
    covered = {(s, t) for s, t in truth_edges if s in represented and t in represented}
    mapped = {(matches[s], matches[t]) for s, t in predicted_edges
              if s in matches and t in matches}
    correct = mapped & truth_edges
    return dict(correct=correct, both_endpoints_present=covered,
                missing_despite_endpoints=covered - correct,
                missing_endpoint=truth_edges - covered)


def compare_support(control, candidate):
    common = control['correct'] & candidate['correct']
    return dict(common_correct=len(common),
                candidate_only_correct=len(candidate['correct'] - common),
                control_only_correct=len(control['correct'] - common),
                diagnostic_oracle_union_correct=len(control['correct'] | candidate['correct']),
                control_missing_despite_endpoints=len(control['missing_despite_endpoints']),
                candidate_missing_despite_endpoints=len(candidate['missing_despite_endpoints']),
                control_missing_endpoint=len(control['missing_endpoint']),
                candidate_missing_endpoint=len(candidate['missing_endpoint']))


def validate(root, raw_root):
    raw = RAW['verify'](raw_root)
    terminal = json.loads((root / 'launcher_terminal.json').read_text())
    manifest = json.loads((root / 'focus_raw_linked_graphs.json').read_text())
    if terminal['status'] != 'completed' or terminal['run_id'] != 'focus-raw-learned-linker-v1':
        raise ValueError('complete linker terminal required')
    if manifest['ground_truth_opened'] is not False or manifest['authorized_for_submission'] is not False:
        raise ValueError('invalid linker authorization')
    if set(manifest['graph_sha256']) != RAW['STEMS']:
        raise ValueError('incomplete graph coverage')
    relative = PurePosixPath(manifest['graph_root']).relative_to('/kaggle/working')
    graph_root = root.joinpath(*relative.parts).resolve()
    if not graph_root.is_relative_to(root.resolve()):
        raise ValueError('graph path escapes output directory')
    import tracksdata as td
    prepared = {}
    for stem, digest in manifest['graph_sha256'].items():
        path = graph_root / f'{stem}.geff'
        if tree_hash(path) != digest:
            raise ValueError('linked graph hash mismatch')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        nodes = {int(r['node_id']): {k: r[k] for k in ('node_id', 't', 'z', 'y', 'x')}
                 for r in graph.node_attrs().iter_rows(named=True)}
        edges = [{k: int(r[k]) for k in ('source_id', 'target_id')}
                 for r in graph.edge_attrs().iter_rows(named=True)]
        _validate_graph(nodes, edges)
        actual = [[r[k] for k in ('t', 'z', 'y', 'x')] for r in nodes.values()]
        with np.load(raw_root / 'raw_detections' / f'{stem}.npz', allow_pickle=False) as data:
            if not np.array_equal(canonical(actual), canonical(data['coords'])):
                raise ValueError('linker removed, added or moved raw detections')
        prepared[stem] = {'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges}
    return prepared, raw


def score(root, raw_root, control_root, truth_root):
    candidate, raw = validate(root, raw_root)
    control = REPLAY['validate_arms'](control_root)
    scorer = REPLAY['load_scorer']('current')
    import tracksdata as td
    from geff import GeffMetadata
    rows = {'control': [], 'raw_linker': []}
    attribution = {}
    for stem in sorted(candidate):
        support = {}
        for arm, payload in [('control', control[stem]['control']), ('raw_linker', candidate[stem])]:
            truth_path = truth_root / f'{stem}.geff'
            truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            graph = REPLAY['prediction_graph'](payload)
            er = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
            keys = td.DEFAULT_ATTR_KEYS
            matches = {r[keys.NODE_ID]: r[keys.MATCHED_NODE_ID]
                       for r in graph.node_attrs().iter_rows(named=True)}
            support[arm] = edge_support(matches,
                [(r[keys.EDGE_SOURCE], r[keys.EDGE_TARGET])
                 for r in graph.edge_attrs().iter_rows(named=True)],
                [(r[keys.EDGE_SOURCE], r[keys.EDGE_TARGET])
                 for r in truth.edge_attrs().iter_rows(named=True)])
            if len(support[arm]['correct']) != er.edge_tp:
                raise ValueError('Diagnostic matched-edge count disagrees with organizer scorer')
            total = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row = dict(scorer.per_sample_metrics(er, total, scorer.node_recall(graph, truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(dict(arm=arm, **row)), flush=True)
        attribution[stem] = compare_support(support['control'], support['raw_linker'])
    summaries = {arm: scorer.summarise(values) for arm, values in rows.items()}
    return dict(per_movie=rows, summaries=summaries, edge_error_attribution=attribution,
        attribution_scope='Diagnostic oracle upper bound only; no edge edits or ensemble authorization',
        by_embryo={arm: {e: scorer.summarise([r for r in values if r['embryo'] == e])
                        for e in ('44b6', '6bba')} for arm, values in rows.items()},
        score_delta=summaries['raw_linker']['score']-summaries['control']['score'],
        raw_terminal_sha256=raw['terminal_sha256'], all_raw_nodes_preserved=True,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        validation_scope='base-training diagnostic', authorized_for_submission=False,
        authorized_for_production_promotion=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--raw-root', type=Path, default=ROOT / '.biohub/cache/kernel-outputs/focus3d-raw-detections-v1')
    parser.add_argument('--control-root', type=Path, default=ROOT / '.biohub/cache/kernel-outputs/focus-bridge-cached-control-v2')
    parser.add_argument('--truth-root', type=Path, default=ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train')
    args = parser.parse_args()
    print(json.dumps(score(args.output_root, args.raw_root, args.control_root, args.truth_root), indent=2))
