"""Freeze all 60 source correction feature/partial-label pairs after source audit.

Does not fit a gate or read selection/validation annotations. Prediction-only
features and partial ground-truth labels are stored separately.
"""
import json
from collections import Counter
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_correction_gate_v1 import components,features,FEATURES,apply_mask
from research.trajectory_correction_supervision_v1 import labels


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();name='trajectory-correction-source-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists(),'Inspect any prior attempt'
    reports={0:ROOT/'reports/experiments/trajectory-event-anchor-source-v1.json',
             1:ROOT/'reports/experiments/trajectory-event-anchor-expanded-source-v1.json'}
    expanded=read(reports[1])
    assert expanded['status']=='expanded_source_anchor_diagnostic_complete','Wait for the existing source audit; do not restart it'
    assert expanded['all_predictions_frozen_before_scoring'] and len(expanded['inference'])==52
    proof=read(ROOT/'.biohub/cache/trajectory-correction-components-v1/RESULT.json')
    assert proof['status']=='source_atomic_corrections_verified'
    assert sha(ROOT/'research/trajectory_correction_gate_v1.py')==proof['helper_sha256']
    target.mkdir();records={};totals=Counter()
    report=dict(status='running',source_only=True,model_fitted=False,gate_threshold_selected=False,
        selection_or_validation_opened=False,features_include_ids=False,feature_names=list(FEATURES),
        source_reports_sha256={str(k):sha(v) for k,v in reports.items()},
        gate_helper_sha256=proof['helper_sha256'],
        label_helper_sha256=sha(ROOT/'research/trajectory_correction_supervision_v1.py'),
        source_sha256=sha(Path(__file__)),prior_backbone_independence_not_established=True,
        labels_are_partial_known_link_changes_not_complete_metric_gains=True,authorized_for_submission=False)
    def persist():
        report.update(records=records,totals=totals,seconds=time.monotonic()-started)
        text=json.dumps(report,indent=2)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        for batch in range(8):
            prefix='trajectory-event-source-v1-b'+str(batch)
            folder=ROOT/'.biohub/cache'/(prefix+'-features');frozen=read(folder/'RESULT.json')
            labels_root=ROOT/'.biohub/cache'/(prefix+'-supervision');supervision=read(labels_root/'RESULT.json')
            assert supervision['source_only'] and supervision['feature_receipt_sha256']==sha(folder/'RESULT.json')
            scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan'/('batch-'+str(batch))/'MOVIES.json'
            assert sha(scope)==frozen['source_scope_sha256']==supervision['source_scope_sha256']
            assert all(m['role']=='optimization' for m in read(scope)['movies'])
            base=ROOT/'.biohub/cache'/(prefix+'-full-output')
            backup=read(ROOT/'reports/experiments'/(prefix+'-full-harvest.json'))
            assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
            evaluated=read(reports[int(batch>0)])
            assert evaluated['parameter_key']=='anchor' and evaluated['model_sha256']=='508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
            candidate_root=ROOT/'.biohub/cache'/('trajectory-event-anchor-expanded-source-v1' if batch else 'trajectory-event-anchor-source-v1')
            for stem,item in frozen['per_movie'].items():
                assert stem not in records
                pp=folder/(stem+'-prediction.json');gp=folder/(stem+'-groups.npz');fp=folder/(stem+'-features.npz')
                for path,key in ((pp,'prediction'),(gp,'groups'),(fp,'features')):assert sha(path)==item[key+'_sha256']
                ip=base/(stem+'-original/pre-postprocess.json');assert sha(ip)==hashes[ip.relative_to(base).as_posix()]
                cp=candidate_root/(stem+'-prediction.json');assert sha(cp)==evaluated['inference'][stem]['prediction_sha256']
                initial,baseline,candidate,groups=read(ip),read(pp),read(cp),arrays(gp)
                parts=components(initial,baseline,candidate);x=features(parts,groups,arrays(fp)['features'])
                replay=apply_mask(initial,baseline,candidate,np.ones(len(parts),bool));validate_graph(replay,100)
                canonical=lambda g:dict(g,edges=sorted(g['edges'],key=lambda e:(e['source_id'],e['target_id'])))
                assert canonical(replay)==canonical(candidate)
                # The feature matrix is computed before loading cached labels.
                lp=labels_root/(stem+'-labels.npz');assert sha(lp)==supervision['per_movie'][stem]['labels_sha256']
                truth=arrays(lp);supervised=labels(parts,baseline,candidate,groups,truth['target'],truth['safe'])
                y=np.asarray([r['label'] for r in supervised],np.int8)
                output=target/(stem+'-features.npz');np.savez_compressed(output,features=x,feature_names=np.asarray(FEATURES))
                lp_out=target/(stem+'-labels.json');lp_out.write_text(json.dumps(supervised)+'\n',encoding='utf-8')
                parts_out=target/(stem+'-components.json');parts_out.write_text(json.dumps(parts)+'\n',encoding='utf-8')
                counts=dict(components=len(parts),partial_improvement=int((y==1).sum()),partial_regression=int((y==0).sum()),unlabeled_or_neutral=int((y==-1).sum()))
                records[stem]=dict(batch=batch,embryo=stem.split('_')[0],counts=counts,features_sha256=sha(output),labels_sha256=sha(lp_out),
                    components_sha256=sha(parts_out),baseline_sha256=sha(pp),candidate_sha256=sha(cp),
                    candidate_path=cp.relative_to(ROOT).as_posix(),baseline_path=pp.relative_to(ROOT).as_posix(),initial_path=ip.relative_to(ROOT).as_posix())
                totals.update(counts);persist();print(json.dumps(dict(stem=stem,**counts)),flush=True)
        assert len(records)==60
        report['status']='source_correction_dataset_frozen';persist()
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':main()
