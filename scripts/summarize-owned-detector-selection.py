"""Predeclared complete-movie comparison of two trained arms and frozen D4."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = 'db9d75ad43a9bc74d3f38f3aef48e5e617510a6abb93205a0387e726415d080f'
INITIAL_SHA = '76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
FLOW_SHA = '3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788'
FIELDS = ('score', 'edge_jaccard', 'node_recall', 'division_jaccard')
COUNTS = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes')


def difference(candidate, control):
    rows, ref = candidate['per_movie'], control['per_movie']
    if [r['stem'] for r in rows] != [r['stem'] for r in ref]:
        raise ValueError('Different comparison movies')
    delta = {k: candidate['summary'][k]-control['summary'][k] for k in FIELDS}
    movies = [dict(stem=r['stem'], adjusted_edge_delta=r['adj_edge_jaccard']-b['adj_edge_jaccard'],
                   recall_delta=r['node_recall']-b['node_recall']) for r,b in zip(rows,ref)]
    worst_delta = min(r['adj_edge_jaccard'] for r in rows)-min(r['adj_edge_jaccard'] for r in ref)
    passes = (delta['score']>0 and delta['edge_jaccard']>0 and delta['node_recall']>=-.005-1e-12
              and sum(r['adjusted_edge_delta']>0 for r in movies)>=5
              and min(r['adjusted_edge_delta'] for r in movies)>=-.02-1e-12
              and worst_delta>=-.01-1e-12)
    return dict(summary_deltas=delta, per_movie=movies, worst_movie_score_delta=worst_delta,
                regressions=[r['stem'] for r in movies if r['adjusted_edge_delta'] < -1e-12],
                passes_predeclared_source_gate=passes)


def compare(scores, manifests, pair, baseline, split):
    if (set(scores)!= {'sparse','pu'} or set(manifests)!=set(scores)
        or pair['status']!='verified_detector_fit_pair_not_selection'
        or pair['paired_inputs_identical'] is not True or pair['paired_targets_identical'] is not True
        or pair['authorized_for_submission'] is not False
        or baseline['status']!='verified_detector_tta_selection'
        or baseline['result']['checkpoint_sha256']!=INITIAL_SHA):
        raise ValueError('Verified paired training and frozen strongest baseline required')
    stems = split['folds'][0]['selection']
    reference = baseline['result']
    if len(stems)!=8 or [r['stem'] for r in reference['per_movie']]!=stems:
        raise ValueError('Exact eight-movie strongest reference required')
    for arm in ('sparse','pu'):
        score, manifest = scores[arm], manifests[arm]
        receipt = manifest.get('owned_detector_fit',{})
        tta = manifest.get('detector_spatial_tta',{})
        policy = dict(neural_weight=0., spatial_weight=1., null_logit=-4.5,flow_checkpoint_sha256=FLOW_SHA)
        if (score['status']!='scored_complete_selection' or manifest['status']!='completed'
            or score['checkpoint_sha256']!=pair['arms'][arm]['checkpoint_sha256']
            or manifest['checkpoint_sha256']!=score['checkpoint_sha256']
            or receipt.get('objective')!=arm or receipt.get('steps')!=1000
            or receipt.get('initialization_sha256')!=INITIAL_SHA
            or manifest.get('standalone_image_flow')!=policy or tta.get('views')!=8
            or tta.get('features')!='native unchanged' or tta.get('encode_calls')!=792
            or [r['stem'] for r in score['per_movie']]!=stems
            or [r['stem'] for r in manifest['records']]!=stems
            or any(r['processed_frames']!=100 or r['image_shape'][0]!=100 for r in manifest['records'])
            or any(score[k] is not False for k in ('target_audit_opened','authorized_for_submission'))
            or any(manifest[k] is not False for k in ('target_audit_opened','ground_truth_opened','authorized_for_submission'))
            or any(not math.isfinite(score['summary'][k]) for k in FIELDS)):
            raise ValueError('Complete registered source-only detector arm with fixed linking required')
    versus_base = {arm:difference(scores[arm],reference) for arm in scores}
    paired = difference(scores['pu'],scores['sparse'])
    eligible = []
    if versus_base['sparse']['passes_predeclared_source_gate']: eligible.append('sparse')
    if versus_base['pu']['passes_predeclared_source_gate'] and paired['passes_predeclared_source_gate']: eligible.append('pu')
    chosen = max(eligible,key=lambda arm:scores[arm]['summary']['score']) if eligible else None
    all_scores = dict(frozen=reference,**scores)
    return dict(status='verified_owned_detector_source_comparison', scores=scores,
        versus_frozen_d4=versus_base,pu_versus_sparse=paired,
        counts={arm:{k:sum(r[k] for r in s['per_movie']) for k in COUNTS} for arm,s in all_scores.items()},
        worst_movies={arm:min(s['per_movie'],key=lambda r:r['adj_edge_jaccard']) for arm,s in all_scores.items()},
        chosen_arm_for_further_evaluation=chosen,
        decision='source_gain_requires_independent_embryo_evaluation' if chosen else 'no_robust_source_selection_gain',
        gate='Score and raw edge Jaccard gain; mean recall loss <=0.005; >=5/8 improved movies; each adjusted-edge loss <=0.02; worst-movie loss <=0.01. PU must also beat trained sparse control.',
        target_audit_opened=False,authorized_for_submission=False,
        caveat='These repeatedly consulted source-selection movies are development evidence, not an unbiased estimate of private-test score; remaining target movies stay closed until a separate frozen audit is declared.')


if __name__=='__main__':
    paths = dict(pair=ROOT/'reports/experiments/owned-detector-fit-pair-v1-result.json',
                 baseline=ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json',
                 split=ROOT/'research/independent_real_baseline_v1_split.json')
    if hashlib.sha256(paths['baseline'].read_bytes()).hexdigest()!=BASE_SHA:
        raise ValueError('Strongest frozen baseline changed')
    scores, manifests = {}, {}
    for arm in ('sparse','pu'):
        paths[arm+'_score'] = ROOT/f'.biohub/cache/kernel-outputs/owned-detector-{arm}-scoring-v1/owned_detector_{arm}_score/selection_score.json'
        paths[arm+'_manifest'] = ROOT/f'.biohub/cache/kernel-outputs/owned-detector-{arm}-selection-v1/owned_detector_{arm}_selection/outputs/selection_manifest.json'
        scores[arm] = json.loads(paths[arm+'_score'].read_text())
        manifests[arm] = json.loads(paths[arm+'_manifest'].read_text())
    report = compare(scores,manifests,*(json.loads(paths[k].read_text()) for k in ('pair','baseline','split')))
    report['source_sha256'] = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/owned-detector-selection-v1-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','decision','chosen_arm_for_further_evaluation','versus_frozen_d4','pu_versus_sparse')},indent=2))
