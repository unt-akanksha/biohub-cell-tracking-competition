"""Record embryo-held-out evidence without calling it leaderboard performance."""
import hashlib
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/embryo_audit_contract.py'))


def summarize(result,manifest,split):
    contract=G['contract'](split)
    if (result['status']!='scored_complete_embryo_audit' or result.get('embryo_audit')!=contract
        or manifest.get('embryo_audit')!=contract or manifest['status']!='completed'
        or result['checkpoint_sha256']!=G['CHECKPOINT_SHA'] or manifest['checkpoint_sha256']!=G['CHECKPOINT_SHA']
        or any(v['target_audit_opened'] is not True or v['authorized_for_submission'] is not False for v in (result,manifest))
        or manifest['ground_truth_opened'] is not False):
        raise ValueError('Explicit completed frozen embryo audit required')
    rows=result['per_movie']
    if ([r['stem'] for r in rows]!=contract['stems'] or [r['stem'] for r in manifest['records']]!=contract['stems']
        or any(r['processed_frames']!=r['image_shape'][0] for r in manifest['records'])
        or result['summary']['n']!=4 or set(result['by_embryo'])!={'44b6'}):
        raise ValueError('Exactly four complete target-embryo movies required')
    return dict(status='verified_initial_embryo_audit',result=result,
        counts={key:sum(r[key] for r in rows) for key in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes')},
        worst_movie=min(rows,key=lambda r:r['adj_edge_jaccard']),
        unaudited_target_movies=len(split['folds'][0]['audit_order'])-len(rows),
        authorized_for_submission=False,
        caveat='Frozen first-four complete embryo-held-out audit, not full target69, reciprocal fold, public-best comparison or hidden-test runtime verification')


if __name__=='__main__':
    paths=dict(result=ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-audit-scoring-v1/detector_spatial_tta_audit_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-audit-v1/detector_spatial_tta_audit/outputs/selection_manifest.json')
    report=summarize(*(json.loads(paths[k].read_text()) for k in ('result','manifest')),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()))
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/detector-spatial-tta-audit-v1-score.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status=report['status'],summary=report['result']['summary'],counts=report['counts'],
        worst_movie=report['worst_movie'],unaudited_target_movies=report['unaudited_target_movies']),indent=2))
