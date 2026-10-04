"""Read-only error attribution on the already-opened eight source movies.

Uses the pinned official node matching; does not tune or rewrite any graph.
An unmatched prediction is not automatically a false detection under sparse GT.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE_SHA = 'db9d75ad43a9bc74d3f38f3aef48e5e617510a6abb93205a0387e726415d080f'


def decompose_edges(gt_edges, node_mapping, pred_edges):
    truth = set(tuple(map(int,e)) for e in gt_edges)
    if len(truth)!=len(gt_edges): raise ValueError('Duplicate ground-truth edges')
    mapping = {int(k):int(v) for k,v in node_mapping.items() if v is not None and int(v)>=0}
    detected = set(mapping.values())
    recovered = {(mapping[a],mapping[b]) for a,b in pred_edges if a in mapping and b in mapping} & truth
    missed = truth-recovered
    counts = dict(recovered=len(recovered),missing_both_endpoints=0,missing_source_only=0,
                  missing_target_only=0,both_endpoints_detected_but_not_linked=0)
    for source,target in missed:
        a,b = source in detected,target in detected
        if not a and not b: counts['missing_both_endpoints']+=1
        elif not a: counts['missing_source_only']+=1
        elif not b: counts['missing_target_only']+=1
        else: counts['both_endpoints_detected_but_not_linked']+=1
    if sum(counts.values())!=len(truth): raise ValueError('Edge attribution is not exhaustive')
    counts['unrecoverable_without_detection_change'] = sum(counts[k] for k in
        ('missing_both_endpoints','missing_source_only','missing_target_only'))
    counts['total_gt_edges'] = len(truth)
    counts['association_recall_given_both_detected'] = len(recovered)/(len(recovered)+counts['both_endpoints_detected_but_not_linked']) if len(recovered)+counts['both_endpoints_detected_but_not_linked'] else None
    return counts


def main():
    import tracksdata as td
    from research.empty_graph_schema import restore_empty_spatial_schema
    scorer_module = runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    scorer = scorer_module['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    infer = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))
    report_path = ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json'
    if hashlib.sha256(report_path.read_bytes()).hexdigest()!=BASE_SHA:
        raise ValueError('Frozen baseline report changed')
    baseline = json.loads(report_path.read_text())['result']
    cache = ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-selection-v1/detector_spatial_tta_selection/outputs'
    manifest = json.loads((cache/'selection_manifest.json').read_text())
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    stems = split['folds'][0]['selection']
    if ([r['stem'] for r in manifest['records']]!=stems
        or [r['stem'] for r in baseline['per_movie']]!=stems or len(stems)!=8):
        raise ValueError('Source-selection scope changed')
    rows = []
    for record, expected in zip(manifest['records'],baseline['per_movie']):
        stem = record['stem']; path = cache/(stem+'.geff')
        if infer['tree_hash'](path)!=record['graph_sha256']:
            raise ValueError('Frozen graph missing or changed: '+stem)
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        truth = td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
        restore_empty_spatial_schema(graph)
        result = scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        if any(getattr(result,k)!=expected[k] for k in result._fields):
            raise ValueError('Local official counts differ from frozen CPU score: '+stem)
        mapping = dict(graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]).select(
            td.DEFAULT_ATTR_KEYS.NODE_ID,td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).iter_rows())
        gt_edges = list(truth.edge_attrs().select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET).iter_rows())
        pred_edges = list(graph.edge_attrs().select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET).iter_rows())
        counts = decompose_edges(gt_edges,mapping,pred_edges)
        if counts['recovered']!=result.edge_tp or counts['total_gt_edges']-counts['recovered']!=result.edge_fn:
            raise ValueError('Diagnostic decomposition differs from official matched edges')
        rows.append(dict(stem=stem,embryo=stem.split('_')[0],**counts,official_edge_fp=result.edge_fp,
            node_recall=expected['node_recall'],graph_sha256=record['graph_sha256']))
        print(json.dumps(rows[-1]),flush=True)
    integer_fields = [k for k,v in rows[0].items() if isinstance(v,int)]
    totals = {k:sum(r[k] for r in rows) for k in integer_fields}
    output = dict(status='verified_frozen_source_error_attribution',baseline_report_sha256=BASE_SHA,
        per_movie=rows,totals=totals,target_audit_opened=False,graphs_modified=False,
        authorized_for_submission=False,
        caveat='Already-consulted source development movies only; matched-endpoint availability is not proof that a linker can recover an edge. Sparse unannotated predictions are not labelled false detections.')
    (ROOT/'reports/experiments/frozen-detector-selection-error-attribution.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(status=output['status'],totals=totals)),flush=True)


if __name__=='__main__': main()
