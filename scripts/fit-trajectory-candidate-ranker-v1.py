"""One fixed source-only LOMO/embryo-transfer diagnostic; not a submission."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_candidate_ranker_v1 import FEATURES, features, supervised, fit, evaluate


def main():
    started=time.monotonic()
    target=ROOT/'.biohub/cache/trajectory-candidate-ranker-v1'
    assert not target.exists()
    inventory=ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json'
    proof=json.loads(inventory.read_text())
    assert proof['status']=='source_candidate_inventory_complete' and proof['source_only']
    plan=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan)==proof['source_scope_sha256']=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    stems=[m['stem'] for m in json.loads(plan.read_text())['movies']]
    base=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    receipt=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert receipt['status']=='verified_backup'
    for r in receipt['records']:assert sha(base/r['path'])==r['sha256']
    data=ROOT/'.biohub/cache/trajectory-candidate-supervision-v1'
    target.mkdir()
    contract=dict(feature_names=FEATURES,regularization=.01,max_iterations=300,
                  source_inventory_sha256=sha(inventory),source_scope_sha256=sha(plan),
                  helper_sha256=sha(ROOT/'research/trajectory_candidate_ranker_v1.py'),
                  source_sha256=sha(Path(__file__)),
                  selection='leave one whole movie out within each source embryo, no hyperparameter search',
                  gate='per embryo pooled learned correct strictly exceeds both current and initial; no held-movie loss versus current; opposite embryo also strictly exceeds both',
                  prediction_changes_forbidden=True,public_backbone_training_overlap=True)
    (target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    rows={};frozen={}
    for stem in stems:
        path=data/(stem+'-candidates.npz');label_path=data/(stem+'-labels.npz')
        assert sha(path)==proof['candidate_sha256'][stem] and sha(label_path)==proof['label_sha256'][stem]
        with np.load(path,allow_pickle=False) as f:groups=dict(f)
        folder=base/(stem+'-original')
        initial=json.loads((folder/'pre-postprocess.json').read_text())
        final=json.loads((folder/'repaired-prediction.json').read_text())
        with np.load(folder/'raw-candidates.npz',allow_pickle=False) as raw:
            matrix=features(initial,final,groups,raw['edges'])
        feature_path=target/(stem+'-features.npz')
        np.savez_compressed(feature_path,features=matrix,feature_names=np.asarray(FEATURES))
        frozen[stem]=sha(feature_path)
        with np.load(label_path,allow_pickle=False) as labels:
            rows[stem]=supervised(matrix,groups,labels['target'],labels['safe'])
        print(json.dumps(dict(stem=stem,stage='features_prepared',queries=len(rows[stem]))),flush=True)
    (target/'PREFIT.json').write_text(json.dumps(frozen,indent=2)+'\n')
    lomo={};source={};cross={};weights={}
    for embryo in ('44b6','6bba'):
        members=[s for s in stems if s.startswith(embryo+'_')]
        for held in members:
            train=[r for s in members if s!=held for r in rows[s]]
            w,fit_proof=fit(train)
            lomo[held]=dict(evaluate(rows[held],w),fit=fit_proof)
            print(json.dumps(dict(stem=held,stage='held_movie',**lomo[held])),flush=True)
        pooled={k:sum(lomo[s][k] for s in members) for k in ('queries','learned_correct','current_correct','initial_correct','learned_repairs','learned_breaks')}
        pooled['passed']=bool(pooled['learned_correct']>max(pooled['current_correct'],pooled['initial_correct']) and all(lomo[s]['learned_correct']>=lomo[s]['current_correct'] for s in members))
        source[embryo]=pooled
        weights[embryo],fit_proof=fit([r for s in members for r in rows[s]])
        held_rows=[r for s in stems if not s.startswith(embryo+'_') for r in rows[s]]
        cross[embryo]=dict(evaluate(held_rows,weights[embryo]),fit=fit_proof)
        cross[embryo]['passed']=cross[embryo]['learned_correct']>max(cross[embryo]['current_correct'],cross[embryo]['initial_correct'])
    qualified=[e for e in source if source[e]['passed'] and cross[e]['passed']]
    np.savez_compressed(target/'weights.npz',**weights,feature_names=np.asarray(FEATURES))
    result=dict(status='source_ranker_diagnostic_complete',contract_sha256=sha(target/'CONTRACT.json'),
                features_sha256=frozen,weights_sha256=sha(target/'weights.npz'),lomo=lomo,
                source=source,cross_embryo=cross,qualified_for_independent_selection=qualified,
                authorized_for_submission=False,predictions_changed=False,hyperparameter_search=False,
                new_selection_or_validation_read=False,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-candidate-ranker-v1-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],source=source,cross=cross,qualified=qualified,seconds=result['seconds'])))


if __name__=='__main__':main()
