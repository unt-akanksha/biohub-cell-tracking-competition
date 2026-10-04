"""Verify frozen transfer evidence; a diagnostic pass is not submission approval."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
G=runpy.run_path(str(ROOT/'research/flow_transfer_contract.py'))
BASE_SHA='d59568f8f69b700a29af62f128cac27ca1b524b2a57f8cad1c7e403d6d6890fa'


def compare(score,manifest,baseline,policy):
    reference=baseline['result']
    if (score['status']!='scored_exposed_flow_transfer_diagnostic'
        or score['source_manifest_run_id']!='flow-tta-transfer-v1'
        or score['flow_transfer_diagnostic']!=policy
        or manifest['flow_spatial_tta']['contract']!=policy
        or any(x['checkpoint_sha256']!=policy['checkpoint_sha256'] for x in (score,manifest))
        or any(x['target_audit_opened'] is not True or x['authorized_for_submission'] is not False for x in (score,manifest))
        or any([r['stem'] for r in rows]!=G['STEMS'] for rows in (score['per_movie'],reference['per_movie'],manifest['records']))
        or any(r['processed_frames']!=100 or r['image_shape'][0]!=100 for r in manifest['records'])):
        raise ValueError('Exact complete previously-exposed transfer evidence required')
    fields=('score','edge_jaccard','node_recall')
    if any(not math.isfinite(x['summary'][k]) for x in (score,reference) for k in fields):
        raise ValueError('Finite aggregate comparison required')
    rows=[]
    for current,old in zip(score['per_movie'],reference['per_movie']):
        if (current['num_pred_nodes']!=old['num_pred_nodes']
            or current['node_recall']!=old['node_recall']
            or not all(math.isfinite(x['adj_edge_jaccard']) for x in (current,old))):
            raise ValueError('Same detection evidence and finite movie scores required')
        rows.append(dict(stem=current['stem'],adjusted_edge_delta=current['adj_edge_jaccard']-old['adj_edge_jaccard']))
    delta={k:score['summary'][k]-reference['summary'][k] for k in fields}
    worst=min(r['adj_edge_jaccard'] for r in score['per_movie'])-min(r['adj_edge_jaccard'] for r in reference['per_movie'])
    conditions=dict(score_gain=delta['score']>0,raw_edge_gain=delta['edge_jaccard']>0,
        mean_recall_preserved=delta['node_recall']>=-.005-1e-12,
        three_movies_improve=sum(r['adjusted_edge_delta']>0 for r in rows)>=3,
        per_movie_loss_bounded=min(r['adjusted_edge_delta'] for r in rows)>=-.02-1e-12,
        worst_movie_preserved=worst>=-.01-1e-12)
    counts=('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes')
    return dict(status='verified_flow_tta_exposed_transfer',result=score,summary_deltas=delta,
        per_movie_deltas=rows,worst_movie_delta=worst,diagnostic_conditions=conditions,
        diagnostic_transfer_supported=all(conditions.values()),
        counts={name:{k:sum(r[k] for r in result['per_movie']) for k in counts}
            for name,result in (('baseline',reference),('flow_d4',score))},
        source_gate_passed=True,authorized_for_submission=False,new_target_movies_opened=0,
        caveat='Previously exposed target movies: transfer diagnostic only, not fresh independent confirmation or a leaderboard score.')


def main():
    paths=dict(source=ROOT/'reports/experiments/flow-spatial-tta-selection-v1-result.json',
        probe=ROOT/'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json',
        split=ROOT/'research/independent_real_baseline_v1_split.json',
        baseline=ROOT/'reports/experiments/detector-spatial-tta-audit-v1-score.json',
        notebook=ROOT/'kaggle/biohub-flow-tta-transfer-v1/biohub-flow-tta-transfer-v1.ipynb',
        manifest=ROOT/'.biohub/cache/kernel-outputs/flow-tta-transfer-v1/flow_tta_transfer/outputs/selection_manifest.json',
        terminal=ROOT/'.biohub/cache/kernel-outputs/flow-tta-transfer-v1/flow_tta_transfer/launcher_terminal.json',
        score=ROOT/'.biohub/cache/kernel-outputs/flow-tta-transfer-scoring-v1/flow_tta_transfer_score/selection_score.json')
    policy=G['verify'](*(paths[k].read_bytes() for k in ('source','probe','split')))
    if hashlib.sha256(paths['baseline'].read_bytes()).hexdigest()!=BASE_SHA:
        raise ValueError('Frozen target baseline changed')
    nb,manifest,terminal,split,score,base=(json.loads(paths[k].read_text()) for k in
        ('notebook','manifest','terminal','split','score','baseline'))
    if nb['metadata']['codex']['flow_spatial_tta']!=policy: raise ValueError('Different frozen motion transfer')
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['verify_manifest'](manifest,terminal,nb['metadata']['codex'],split)
    helper=runpy.run_path(str(ROOT/'scripts/run-owned-detector-evaluation-queue.py'))
    ref='indarkarhana/biohub-flow-tta-transfer-scoring-v1'
    remote=helper['INSPECT'](ref)
    if (not remote['present'] or remote['current_version_number']!=1
        or helper['status_value'](helper['command'](['kaggle','kernels','status',ref]),ref)!='COMPLETE'):
        raise ValueError('Exact completed CPU scoring version required')
    result=compare(score,manifest,base,policy)
    result.update(cpu_kernel_ref=ref+'/1',cpu_status='COMPLETE',
        source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
    destination=ROOT/'reports/experiments/flow-tta-transfer-v1-result.json'
    if destination.exists(): raise ValueError('Refuse to overwrite completed transfer comparison')
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='result'},indent=2))


if __name__=='__main__': main()
