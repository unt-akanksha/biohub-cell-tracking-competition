"""Fixed three-epoch source6 fit on batch 0; hold source44 and batch 1 out."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_case_store_v1 import CaseStore
from research.trajectory_event_features_v1 import FEATURES
from research.trajectory_event_training_v1 import prior,fit


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    started=time.monotonic();target=ROOT/'.biohub/cache/trajectory-event-source-pilot-v1'
    receipt=ROOT/'reports/experiments/trajectory-event-source-pilot-v1.json'
    assert not target.exists() and not receipt.exists(), 'Inspect the existing run, never silently restart'
    cases_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-cases';manifest=read(cases_root/'RESULT.json')
    assert manifest['status']=='source_event_cases_prepared' and manifest['training_allowed']
    assert manifest['source_only'] and not manifest['selection_or_validation_opened']
    scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan/batch-0/MOVIES.json'
    plan=read(scope);assert sha(scope)==manifest['source_scope_sha256']
    assert all(m['role']=='optimization' for m in plan['movies'])
    for name,digest in manifest['helper_sha256'].items():assert sha(ROOT/'research'/name)==digest
    labels=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-supervision/RESULT.json'
    assert sha(labels)==manifest['supervision_receipt_sha256']
    stems=[m['stem'] for m in plan['movies'] if m['stem'].startswith('6bba_')]
    assert len(stems)==7
    records=[];divisions=0;opportunities=0
    for stem in stems:
        row=manifest['per_movie'][stem]
        assert not row['counts'].get('incompatible_partial_constraints',0)
        records.extend(dict(r,stem=stem) for r in row['cases'] if r['reason']=='prepared')
        divisions+=row['source_prior_counts']['annotated_division_parents']
        opportunities+=row['source_prior_counts']['annotated_consecutive_parent_opportunities']
    model=ROOT/'.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
    assert sha(model)=='e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
    with np.load(model,allow_pickle=False) as values:anchor=prior(values['6bba'],divisions,opportunities)
    penalty=np.r_[np.full(18,.1),np.full(12,.02)]
    class DeadlineStore(CaseStore):
        def __getitem__(self,index):
            if time.monotonic()-started>1200:raise TimeoutError('Twenty-minute CPU fit deadline; completed epochs preserved')
            return super().__getitem__(index)
    store=DeadlineStore(cases_root,records,cache_size=2)
    target.mkdir()
    contract=dict(source_stems=stems,source_embryo='6bba',source_scope_sha256=sha(scope),
        case_manifest_sha256=sha(cases_root/'RESULT.json'),cases=len(records),
        source_prior_counts=dict(divisions=divisions,opportunities=opportunities),
        anchor=anchor.tolist(),regularization=penalty.tolist(),epochs=3,learning_rate=.03,seed=20260914,
        exact_oracle_seconds=5.,cpu_deadline_seconds=1200,feature_names=list(FEATURES),
        initializer_sha256=sha(model),initializer_is_smoke_checkpoint=False,
        source_sha256=sha(Path(__file__)),
        modules_sha256={n:sha(ROOT/'research'/n) for n in ('trajectory_event_training_v1.py',
            'trajectory_event_assignment_v1.py','trajectory_event_case_store_v1.py')},
        event_head_held_out_scope='source44 batch0 and all eight batch1 source movies',
        no_source_score_based_epoch_selection=True,selection_or_validation_used_for_training=False,
        source_only=True,authorized_for_submission=False)
    (target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    report=dict(status='running',contract_sha256=sha(target/'CONTRACT.json'),epochs_completed=[],
                source_only=True,authorized_for_submission=False,quality_gain_established=False)
    def persist():
        report['seconds']=time.monotonic()-started
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        # Real serialized case smoke before the full traversal/optimizer.
        for index in (0,len(store)-1):
            assert store[index]['x'].shape[1]==30 and store[index]['n_constraints']>0
        report['serialized_case_smoke_passed']=True;persist()
        print(json.dumps(dict(event='source_fit_started',cases=len(store),movies=len(stems),divisions=divisions,opportunities=opportunities,maximum_seconds=1200)),flush=True)
        def checkpoint(weights,row):
            path=target/('epoch-'+str(row['epoch'])+'.npz')
            np.savez_compressed(path,weights=weights,anchor=anchor,regularization=penalty,feature_names=np.asarray(FEATURES))
            row.update(weights_sha256=sha(path),elapsed_seconds=time.monotonic()-started)
            report['epochs_completed'].append(row);persist()
            print(json.dumps(dict(event='epoch_complete',**row)),flush=True)
        weights,history=fit(store,anchor,penalty,epochs=3,learning_rate=.03,seed=20260914,time_limit=5.,callback=checkpoint)
        assert len(history)==3 and np.isfinite(weights).all()
        report.update(status='source_event_pilot_training_complete',weights_sha256=sha(target/'epoch-3.npz'),
                      weights_changed=not np.array_equal(weights,anchor),model_fitted=True,
                      source_sha256=sha(Path(__file__)))
        persist();print(json.dumps(dict(status=report['status'],weights_sha256=report['weights_sha256'],seconds=report['seconds'])),flush=True)
    except BaseException as error:
        report.update(status='timeout' if isinstance(error,TimeoutError) else 'failed',error=repr(error));persist();raise


if __name__=='__main__':main()
