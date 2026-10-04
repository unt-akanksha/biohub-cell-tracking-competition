"""Real source-only replay and selective-mask invariant tests; no gate fitting."""
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_correction_gate_v1 import components,apply_mask,features,FEATURES


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();name='trajectory-correction-components-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    prior=ROOT/'reports/experiments/trajectory-event-anchor-source-v1.json'
    assert sha(prior)=='e1667c04fd1dd83d18932f63981e98e54ff0a2f72873932f55216987131b44fd'
    source=read(prior);folder=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features'
    frozen=read(folder/'RESULT.json');base=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-full-output'
    backup=read(ROOT/'reports/experiments/trajectory-event-source-v1-b0-full-harvest.json')
    assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
    candidate_root=ROOT/'.biohub/cache/trajectory-event-anchor-source-v1'
    records={};target.mkdir();rng=np.random.default_rng(20260914)
    for stem in frozen['per_movie']:
        item=frozen['per_movie'][stem]
        pp=folder/(stem+'-prediction.json');gp=folder/(stem+'-groups.npz');fp=folder/(stem+'-features.npz')
        for p,key in ((pp,'prediction'),(gp,'groups'),(fp,'features')):assert sha(p)==item[key+'_sha256']
        ip=base/(stem+'-original/pre-postprocess.json');assert sha(ip)==hashes[ip.relative_to(base).as_posix()]
        cp=candidate_root/(stem+'-prediction.json')
        assert sha(cp)==source['inference'][stem]['prediction_sha256']
        initial,baseline,candidate=read(ip),read(pp),read(cp)
        parts=components(initial,baseline,candidate);x=features(parts,arrays(gp),arrays(fp)['features'])
        for selected,expected in ((False,baseline),(True,candidate)):
            replay=apply_mask(initial,baseline,candidate,np.full(len(parts),selected,dtype=bool))
            # Edge ordering is not semantically relevant; all edge attributes are.
            canonical=lambda g:dict(g,edges=sorted(g['edges'],key=lambda e:(e['source_id'],e['target_id'])))
            assert canonical(replay)==canonical(expected)
        for _ in range(3):
            mixed=apply_mask(initial,baseline,candidate,rng.random(len(parts))>.5)
            validate_graph(mixed,100);assert mixed['nodes']==baseline['nodes']
        archive=target/(stem+'-features.npz');np.savez_compressed(archive,features=x,feature_names=np.asarray(FEATURES))
        scope=target/(stem+'-components.json');scope.write_text(json.dumps(parts)+'\n',encoding='utf-8')
        records[stem]=dict(components=len(parts),removed=sum(len(p['removed']) for p in parts),
            added=sum(len(p['added']) for p in parts),features_sha256=sha(archive),components_sha256=sha(scope),
            candidate_sha256=sha(cp),baseline_sha256=sha(pp),exact_all_and_none_replay=True,random_valid_masks=3)
        print(json.dumps(dict(stem=stem,**records[stem])),flush=True)
    report=dict(status='source_atomic_corrections_verified',records=records,feature_names=list(FEATURES),
        source_only=True,annotations_opened=False,model_fitted=False,gate_threshold_selected=False,
        candidate_changed=False,source_result_sha256=sha(prior),source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT/'research/trajectory_correction_gate_v1.py'),seconds=time.monotonic()-started)
    text=json.dumps(report,indent=2)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],movies=len(records),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
