"""Freeze label-free event inputs from the accepted eight-movie T4 cache."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_candidate_ranker_v1 import FEATURES, features


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    start=time.monotonic();name='trajectory-event-eight-v1-features'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    prior=ROOT/'.biohub/cache/trajectory-structured-eight-v1'
    assert sha(prior/'RESULT.json')=='0469b99ccf3d347a72ec00afb7206ea8fa6bd6498a7c388e09b0d8cf667508ce'
    old=read(prior/'RESULT.json');assert old['status']=='complete_eight_movie_validation'
    accepted_path=ROOT/'reports/experiments/trajectory-overlap-cache-kaggle-v2-result.json'
    assert sha(accepted_path)=='2934de829c237906f214f7c6e4583b961b6e59aeb73c8a8adde05c52d69a1ede'
    accepted=read(accepted_path);assert accepted['status']=='acceptance_passed'
    cache=ROOT/'.biohub/cache/trajectory-overlap-cache-kaggle-v1-output'
    assert sha(cache/'trajectory-complete/result.json')==accepted['terminal_sha256']
    paths={r['movie']:cache/r['path'] for r in accepted['verified_graphs'] if r['arm']=='repaired'}
    assert len(paths)==8 and set(paths)==set(old['records'])
    native=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert sha(native)=='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    assert not set(paths)&{m['stem'] for m in read(native)['movies']}
    model=read(ROOT/'.biohub/cache/trajectory-structured-portable-v1/structured-trajectory-model.json')
    for name in ('trajectory_candidate_inventory_v1.py','trajectory_candidate_ranker_v1.py'):
        assert sha(ROOT/'research'/name)==model['inference_source_sha256'][name]
    target.mkdir();records={}
    for stem,path in paths.items():
        frozen=old['records'][stem]
        pp=prior/(stem+'-prediction.json');ip=path.parent/'pre-postprocess.json';rp=path.parent/'raw-candidates.npz'
        for p,key in ((pp,'prediction'),(ip,'initial'),(rp,'raw'),(path,'baseline')):
            assert sha(p)==frozen[key+'_sha256']
        graph=read(pp);initial=read(ip);validate_graph(graph,100)
        groups=candidates(initial,graph)
        with np.load(rp,allow_pickle=False) as raw:matrix=features(initial,graph,groups,raw['edges'])
        gp=target/(stem+'-groups.npz');fp=target/(stem+'-features.npz');bp=target/(stem+'-prediction.json')
        np.savez_compressed(gp,**groups)
        np.savez_compressed(fp,features=matrix,feature_names=np.asarray(FEATURES))
        bp.write_text(json.dumps(graph,sort_keys=True,allow_nan=False),encoding='utf-8')
        records[stem]=dict(groups_sha256=sha(gp),features_sha256=sha(fp),prediction_sha256=sha(bp),
            previous_prediction_sha256=sha(pp),initial_sha256=sha(ip),raw_sha256=sha(rp),
            initial_path=ip.relative_to(ROOT).as_posix(),queries=len(groups['children']),candidate_edges=len(groups['parents']))
        print(json.dumps(dict(event='validation_inputs_frozen',stem=stem,queries=len(groups['children']))),flush=True)
    report=dict(status='eight_event_inputs_frozen_no_new_labels',per_movie=records,
        native_role_plan_sha256=sha(native),baseline_score_receipt_sha256=sha(prior/'RESULT.json'),
        accepted_T4_cache_sha256=sha(accepted_path),source_sha256=sha(Path(__file__)),
        source_and_selection_role_disjoint=True,model_fitted=False,ground_truth_opened=False,
        validation_labels_used=False,prior_pipeline_validation_exposure_disclosed=True,
        public_backbone_independence_not_established=True,authorized_for_submission=False,
        seconds=time.monotonic()-start)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
