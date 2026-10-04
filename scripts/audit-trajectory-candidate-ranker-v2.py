"""Replay unchanged v1 fits; evaluate all choices without GT candidate masks."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_candidate_ranker_v1 import supervised,fit
from research.trajectory_candidate_ranker_audit_v2 import predict,evaluate


def main():
    started=time.monotonic()
    target=ROOT/'.biohub/cache/trajectory-candidate-ranker-v2-audit'
    assert not target.exists()
    base=ROOT/'.biohub/cache/trajectory-candidate-ranker-v1'
    prior=json.loads((base/'RESULT.json').read_text())
    contract=json.loads((base/'CONTRACT.json').read_text())
    assert sha(base/'CONTRACT.json')==prior['contract_sha256']
    assert sha(ROOT/'research/trajectory_candidate_ranker_v1.py')==contract['helper_sha256']
    inv_path=ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json'
    assert sha(inv_path)==contract['source_inventory_sha256']
    inv=json.loads(inv_path.read_text());data=ROOT/'.biohub/cache/trajectory-candidate-supervision-v1'
    target.mkdir()
    groups={};matrices={};targets={};rows={}
    for stem,digest in prior['features_sha256'].items():
        assert sha(base/(stem+'-features.npz'))==digest
        assert sha(data/(stem+'-candidates.npz'))==inv['candidate_sha256'][stem]
        assert sha(data/(stem+'-labels.npz'))==inv['label_sha256'][stem]
        with np.load(base/(stem+'-features.npz'),allow_pickle=False) as f:matrices[stem]=f['features']
        with np.load(data/(stem+'-candidates.npz'),allow_pickle=False) as f:groups[stem]=dict(f)
        with np.load(data/(stem+'-labels.npz'),allow_pickle=False) as f:
            targets[stem]=f['target']
            rows[stem]=supervised(matrices[stem],groups[stem],targets[stem],f['safe'])
    assert sha(base/'weights.npz')==prior['weights_sha256']
    with np.load(base/'weights.npz',allow_pickle=False) as f:old_weights={e:f[e] for e in ('44b6','6bba')}
    frozen={};lomo={};cross={};source={};weights={}
    for embryo in ('44b6','6bba'):
        members=[s for s in groups if s.startswith(embryo+'_')]
        for held in members:
            w,_=fit([r for s in members if s!=held for r in rows[s]])
            weights['lomo_'+held]=w
            predictions=predict(matrices[held],groups[held],w)
            path=target/(held+'-lomo.npy');np.save(path,predictions,allow_pickle=False);frozen[path.name]=sha(path)
            lomo[held]=evaluate(predictions,groups[held],targets[held])
        summed={k:sum(lomo[s][k] for s in members) for k in ('queries','learned_correct','current_correct','initial_correct','learned_repairs','learned_breaks')}
        summed['passed']=bool(summed['learned_correct']>max(summed['current_correct'],summed['initial_correct']) and all(lomo[s]['learned_correct']>=lomo[s]['current_correct'] for s in members))
        source[embryo]=summed
        w,_=fit([r for s in members for r in rows[s]])
        assert np.array_equal(w,old_weights[embryo]), 'Unchanged fit must exactly reproduce v1 weights'
        weights[embryo]=w;cross_rows={}
        for held in groups:
            if held in members:continue
            predictions=predict(matrices[held],groups[held],w)
            path=target/(held+'-cross.npy');np.save(path,predictions,allow_pickle=False);frozen[path.name]=sha(path)
            cross_rows[held]=evaluate(predictions,groups[held],targets[held])
        summed={k:sum(r[k] for r in cross_rows.values()) for k in ('queries','learned_correct','current_correct','initial_correct','learned_repairs','learned_breaks')}
        summed['passed']=bool(summed['learned_correct']>max(summed['current_correct'],summed['initial_correct']) and all(r['learned_correct']>=r['current_correct'] for r in cross_rows.values()))
        cross[embryo]=dict(summary=summed,per_movie=cross_rows)
    qualified=[e for e in source if source[e]['passed'] and cross[e]['summary']['passed']]
    np.savez_compressed(target/'weights.npz',**weights)
    result=dict(status='unmasked_source_ranker_audit_complete',source=source,lomo=lomo,cross_embryo=cross,
                qualified_for_independent_selection=qualified,prior_sha256=sha(base/'RESULT.json'),
                prediction_sha256=frozen,weights_sha256=sha(target/'weights.npz'),
                helper_sha256=sha(ROOT/'research/trajectory_candidate_ranker_audit_v2.py'),source_sha256=sha(Path(__file__)),
                unchanged_fit_weights=True,annotation_mask_used_for_inference=False,
                graph_assignment_tested=False,authorized_for_submission=False,
                selection_or_validation_opened=False,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/experiments/trajectory-candidate-ranker-v2-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],source=source,cross={e:cross[e]['summary'] for e in cross},qualified=qualified,seconds=result['seconds'])))


if __name__=='__main__':main()
