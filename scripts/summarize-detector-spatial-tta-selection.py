"""Changed-detector comparison: tracking gains, recall, counts and worst movie."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
FLOW_SHA='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788'


def compare(candidate,manifest,flow):
    policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,flow_checkpoint_sha256=FLOW_SHA)
    receipt=manifest.get('detector_spatial_tta',{})
    if (candidate['status']!='scored_complete_selection' or manifest['status']!='completed'
        or candidate['checkpoint_sha256']!=CHECKPOINT_SHA or manifest['checkpoint_sha256']!=CHECKPOINT_SHA
        or flow['flow_checkpoint_sha256']!=FLOW_SHA or manifest.get('standalone_image_flow')!=policy
        or receipt.get('views')!=8 or receipt.get('features')!='native unchanged'
        or receipt.get('maximum_mean_absolute_logit_delta',0)<=0
        or receipt.get('encode_calls')!=sum(r['processed_frames']-1 for r in manifest['records'])
        or any(candidate[k] is not False for k in ('target_audit_opened','authorized_for_submission'))
        or any(manifest[k] is not False for k in ('target_audit_opened','ground_truth_opened','authorized_for_submission'))):
        raise ValueError('Completed detector-only TTA with exact standalone-flow reference required')
    rows,baseline=candidate['per_movie'],flow['per_movie']['motion']
    stems=[r['stem'] for r in baseline]
    if len(stems)!=8 or len(set(stems))!=8 or any([r['stem'] for r in records]!=stems for records in (rows,manifest['records'])):
        raise ValueError('Exact complete eight-movie comparison required')
    for record in manifest['records']:
        if record['processed_frames']!=record['image_shape'][0]:
            raise ValueError('Incomplete movie')
    summary,control=candidate['summary'],flow['summaries']['motion']
    fields=('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes')
    counts={label:{k:sum(r[k] for r in group) for k in fields} for label,group in [('candidate',rows),('flow',baseline)]}
    deltas={name:summary[name]-control[name] for name in ('score','edge_jaccard','node_recall','division_jaccard')}
    robust=deltas['score']>0 and deltas['edge_jaccard']>0 and deltas['node_recall']>=-.005-1e-12
    return dict(status='verified_detector_tta_selection',result=candidate,counts=counts,deltas_vs_flow=deltas,
        predicted_node_count_delta=counts['candidate']['num_pred_nodes']-counts['flow']['num_pred_nodes'],
        per_movie_comparison=[dict(stem=r['stem'],score_delta=r['adj_edge_jaccard']-b['adj_edge_jaccard'],
            recall_delta=r['node_recall']-b['node_recall'],nodes_delta=r['num_pred_nodes']-b['num_pred_nodes']) for r,b in zip(rows,baseline)],
        regressions_vs_flow=[r['stem'] for r,b in zip(rows,baseline) if r['adj_edge_jaccard']<b['adj_edge_jaccard']-1e-12],
        worst_movie=min(rows,key=lambda r:r['adj_edge_jaccard']),reference_summary=control,
        decision='source_selection_gain_requires_embryo_audit' if robust else 'no_robust_source_selection_gain',
        recall_guard='No more than0.005 absolute mean node recall loss; require raw edge Jaccard and score gains',
        caveat='Image-derived detections may change; fixed flow/null/topology, no graph-count tuning; source selection only',
        authorized_for_submission=False)


if __name__=='__main__':
    paths=dict(candidate=ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-scoring-v1/detector_spatial_tta_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-selection-v1/detector_spatial_tta_selection/outputs/selection_manifest.json',
        flow=ROOT/'reports/experiments/backward-flow-selection-v1-score.json')
    values={k:json.loads(p.read_text()) for k,p in paths.items()}; values['flow']=values['flow']['result']
    report=compare(**values)
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','decision','deltas_vs_flow','counts','regressions_vs_flow')},indent=2))
