"""Score archived baseline stages and attribute errors, without candidate tuning."""
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,csv_equivalent_graph
from research.trajectory_stage_audit_v1 import attribute


def main():
    started=time.monotonic()
    output=ROOT/'reports/experiments/trajectory-stage-audit-v1-result.json'
    assert not output.exists()
    base=ROOT/'.biohub/cache/dense-warp-movie-v1-full-output'
    terminal=json.loads((base/'result.json').read_text())
    assert sha(base/'result.json')=='b2e581d751f9b9266c6deba4916155480902644845663d977ac2ff8c6fb70b14'
    assert terminal['status']=='complete_prelabel_predictions'
    receipt=json.loads((ROOT/'reports/experiments/dense-warp-movie-v1-full-harvest.json').read_text())
    assert receipt['status']=='verified_backup'
    for row in receipt['records']:
        assert sha(base/row['path'])==row['sha256']
    filenames={'initial_ilp':'pre-postprocess.json','public_postprocess':'prediction.json','submitted':'repaired-prediction.json'}
    prepared={};pins={}
    for stem in terminal['movies']:
        prepared[stem]={}
        for stage,name in filenames.items():
            path=base/(stem+'-original')/name
            graph=json.loads(path.read_text())
            prepared[stem][stage]=csv_equivalent_graph({int(k):v for k,v in graph['nodes'].items()},graph['edges'],100)
            pins[stem+'/'+stage]=sha(path)
    # Only archived baseline stages are analyzed; failed learned arms are excluded.
    truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=truth_root/'train_geff_cache_manifest.json'
    assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
    records=json.loads(manifest.read_text())['files']
    for stem in prepared:
        selected=[r for r in records if r['relative_path'].startswith(stem+'.geff/')]
        assert len(selected)==21
        for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
    previous=ROOT/'reports/experiments/dense-warp-movies-v1-score.json'
    assert sha(previous)=='74e42215696319936b5d74226bb93df77263e8f7e8186155c258f5a6c09ec61a'
    previous={r['stem']:r for r in json.loads(previous.read_text())['rows']['original']}
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
    import polars as pl
    import tracksdata as td
    from geff import GeffMetadata
    keys=td.DEFAULT_ATTR_KEYS
    rows={a:[] for a in filenames};details={}
    for stem,stages in prepared.items():
        initial={(e['source_id'],e['target_id']) for e in stages['initial_ilp']['edges']}
        details[stem]={}
        for stage,payload in stages.items():
            graph=td.graph.InMemoryGraph()
            for axis in ('z','y','x'):graph.add_node_attr_key(axis,pl.Float64,0.)
            graph.add_node_attr_key('audit_original_id',pl.Int64,-1)
            ids=sorted(payload['nodes'],key=int)
            mapped=graph.bulk_add_nodes([dict({k:payload['nodes'][i][k] for k in ('t','z','y','x')},audit_original_id=int(i)) for i in ids])
            mapping=dict(zip(map(int,ids),mapped))
            graph.bulk_add_edges([dict(source_id=mapping[e['source_id']],target_id=mapping[e['target_id']]) for e in payload['edges']])
            path=truth_root/'train'/(stem+'.geff');truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
            er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            estimated=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            row=dict(scorer.per_sample_metrics(er,estimated,scorer.node_recall(graph,truth)),stem=stem,embryo=stem.split('_')[0])
            if stage=='submitted':
                for key in scorer.COUNT_COLUMNS:assert row[key]==previous[stem][key]
            rows[stage].append(row)
            node_rows={r[keys.NODE_ID]:dict(original=r['audit_original_id'],matched=r[keys.MATCHED_NODE_ID]) for r in graph.node_attrs().iter_rows(named=True)}
            edge_rows=[dict(source=r[keys.EDGE_SOURCE],target=r[keys.EDGE_TARGET],matched=r[keys.MATCHED_EDGE_MASK],valid=r['pred_valid']) for r in scorer._evaluate_matched_graph(graph,truth).iter_rows(named=True)]
            truth_edges=[(r[keys.EDGE_SOURCE],r[keys.EDGE_TARGET]) for r in truth.edge_attrs().iter_rows(named=True)]
            attribution=attribute(node_rows,edge_rows,truth_edges,initial)
            counts=attribution['counts']
            assert sum(v for k,v in counts.items() if k.endswith('_tp'))==er.edge_tp
            assert sum(v for k,v in counts.items() if k.endswith('_fp'))==er.edge_fp
            assert len(attribution['missing_gt_edges'])==er.edge_fn
            details[stem][stage]=attribution
            print(json.dumps(dict(stem=stem,stage=stage,counts=counts,summary=helper['finite'](scorer.summarise([row])))),flush=True)
    summaries={a:scorer.summarise(rs) for a,rs in rows.items()}
    result=dict(status='complete_stage_diagnostic',rows=rows,summaries=summaries,details=details,
                input_sha256=pins,source_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'research/trajectory_stage_audit_v1.py'),
                exposed_diagnostic_only=True,causal_intervention=False,authorized_for_submission=False,
                no_prediction_mutation=True,no_failed_linker_rescue=True,seconds=time.monotonic()-started)
    output.write_text(json.dumps(helper['finite'](result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(helper['finite'](summaries)),flush=True)


if __name__=='__main__':main()
