"""Verify the public test: all three graph stages, CSV, source, and runtime risk.

This verifier cannot submit. Version1's one-hour platform cap is explicitly
ineligible for hidden submission until final-version timeout review.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,assemble_csv
from research.trajectory_structured_release_v1 import check_graph_stages,runtime_evidence
from research.trajectory_event_release_v1 import check_event_stage
from research.submission_sharding import validate_submission_kernel_metadata
KERNEL='indarkarhana/biohub-event-anchor-candidate'
BASE=Path('C:/Users/IndarKumar/AppData/Local/Programs/Python/Python312/python.exe')


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    started=time.monotonic();receipt=ROOT/'reports/experiments/trajectory-event-anchor-production-v1-verification.json'
    assert not receipt.exists(),'Inspect previous verification; do not overwrite it'
    build_path=ROOT/'reports/experiments/trajectory-event-anchor-production-v1-build.json';build=read(build_path)
    launch=read(ROOT/'reports/experiments/trajectory-event-anchor-production-v1-launch.json')
    proof_path=ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-result.json';proof=read(proof_path)
    assert sha(proof_path)==build['acceptance_sha256'] and proof['status']=='acceptance_passed'
    assert launch['kernel_pushed'] and launch['exit_code']==0 and launch['public_test_only'] and launch['platform_timeout_seconds']==3600
    assert launch['notebook_sha256']==build['notebook_sha256']
    stage=ROOT/'.biohub/staging/biohub-event-anchor-candidate-v1';remote=ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-source'
    expected=read(stage/'kernel-metadata.json');observed=read(remote/'kernel-metadata.json');validate_submission_kernel_metadata(expected)
    assert expected['id']==KERNEL and sha(stage/expected['code_file'])==build['notebook_sha256']
    for key in ('id','language','kernel_type','is_private','enable_gpu','enable_tpu','enable_internet','dataset_sources',
                'competition_sources','model_sources','kernel_sources','machine_shape','docker_image'):
        assert expected[key]==observed[key],'Remote metadata differs: '+key
    cells=lambda p:[''.join(c['source']) for c in read(p)['cells'] if c['cell_type']=='code']
    assert cells(stage/expected['code_file'])==cells(remote/observed['code_file'])
    assert sha(stage/'derived-contract.json')==build['contract_sha256']==proof['contract_sha256']
    contract=read(stage/'derived-contract.json');bundle=ROOT/'.biohub/cache/trajectory-event-anchor-kaggle-v1-output/trajectory-runtime-v1'
    names=('trajectory_source_mixture_v1.py','learned_trajectory_endpoint_v1.py','source-44b6-target-6bba.npz',
        'source-6bba-target-44b6.npz','structured-trajectory.py','structured-trajectory-model.json','event-trajectory.py','event-trajectory-model.json')
    for name in names:assert sha(bundle/name)==contract['bundle_sha256'][name],'Replay bundle changed: '+name
    mixture=load_module(bundle/names[0],'event_public_mixture');motion=load_module(bundle/names[1],'event_public_motion')
    motion_models=[]
    for name in names[2:4]:
        with np.load(bundle/name,allow_pickle=False) as data:motion_models.append(dict(data))
    structured=load_module(bundle/'structured-trajectory.py','event_public_structured')
    event=load_module(bundle/'event-trajectory.py','event_public_anchor')
    structured_model=read(bundle/'structured-trajectory-model.json');event_model=read(bundle/'event-trajectory-model.json')
    assert list(structured.FEATURES)==structured_model['features'] and list(event.FEATURES)==event_model['features']
    outputs=ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-output'
    harvest_path=ROOT/'reports/experiments/trajectory-event-anchor-production-v1-harvest.json';harvest=read(harvest_path)
    assert harvest['status']=='required_production_evidence_downloaded' and harvest['kernel']==KERNEL
    for name,item in harvest['files'].items():assert sha(outputs/name)==item['sha256'],'Harvested evidence changed: '+name
    result=dict(status='verifying',submission_performed=False,public_test_only=True,
        final_submission_version_requires_twelve_hour_timeout_review=True,quality_tradeoff_release_review_required=True,
        acceptance_sha256=sha(proof_path),build_sha256=sha(build_path),harvest_sha256=sha(harvest_path),records=[])
    def persist():
        result['seconds']=time.monotonic()-started;receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist();paths={};times=[]
    try:
        for folder,mode,nframes in (('trajectory-smoke','smoke',8),('trajectory-complete','production',100)):
            terminal=read(outputs/folder/'result.json')
            assert terminal['status']=='complete' and terminal['mode']==mode
            assert terminal['contract_sha256']==build['contract_sha256'] and terminal['gpu_count']==2 and terminal['worker_count']==4
            assert not terminal['ground_truth_opened'] and terminal['submission_created']==(mode=='production')
            plans={str(s['shard_index']):s for s in terminal['shards']}
            assert set(plans)==set(terminal['workers'])=={'0','1','2','3'}
            for shard,worker in terminal['workers'].items():
                assert worker['status']==('functionality_passed' if mode=='smoke' else 'complete_prelabel_predictions')
                assert worker['inputs_unchanged'] and not worker['ground_truth_opened'] and 'T4' in worker['device'] and worker['solver_backend']=='SCIP'
                assert set(worker['movies'])==set(plans[shard]['movie_ids'])
                for stem,arms in worker['movies'].items():
                    assert set(arms)=={'original'} and arms['original']['frames']==nframes
                    row=arms['original'];path=outputs/folder/('shard-'+shard)/(stem+'-original')
                    initial=read(path/'pre-postprocess.json');baseline=read(path/'prediction.json')
                    moved=read(path/'motion-repaired-prediction.json');assigned=read(path/'structured-repaired-prediction.json');final=read(path/'repaired-prediction.json')
                    assert sha(path/'prediction.json')==row['prediction_sha256'] and sha(path/'repaired-prediction.json')==row['repaired_sha256']
                    details=read(path/'repair-details.json')
                    check_graph_stages(baseline,moved,assigned,nframes,row['added_edges'],details['structured_assignment'])
                    check_event_stage(initial,assigned,final,details['event_assignment'],frames=nframes)
                    assert not details['event_assignment']['budget_exhausted'] and not details['event_assignment']['solver_fallbacks']
                    with np.load(path/'raw-candidates.npz',allow_pickle=False) as raw:
                        replay,_=mixture.reconnect(baseline,raw['coords'],raw['edges'],set(map(int,initial['nodes'])),motion_models,motion.reconnect)
                        assert replay==moved,'Motion replay differs: '+stem
                        replay,_=structured.refine(initial,moved,raw['coords'],raw['edges'],structured_model['weights'])
                        assert replay==assigned,'Structured replay differs: '+stem
                        replay,replay_details=event.refine(initial,assigned,raw['coords'],raw['edges'],event_model['weights'])
                        assert replay==final and not replay_details['budget_exhausted'] and not replay_details['solver_fallbacks'],'Event replay differs: '+stem
                    result['records'].append(dict(mode=mode,stem=stem,frames=nframes,three_stages_exact=True,final_sha256=row['repaired_sha256']))
                    if mode=='production':
                        assert stem not in paths;paths[stem]=path/'repaired-prediction.json';times.append(row['seconds'])
                    persist();print(json.dumps(dict(mode=mode,stem=stem,three_stages_exact=True)),flush=True)
        assert set(paths)=={'44b6_0113de3b','44b6_0b24845f','6bba_05b6850b','6bba_05db0fb1'}
        terminal=read(outputs/'trajectory-complete/result.json');csv_info=terminal['csv']
        assert set(paths)==set(csv_info['movie_rows']) and 0<terminal['elapsed_seconds']<=3600
        verification=ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-verification';verification.mkdir(exist_ok=False)
        csv=verification/'submission-replay.csv';rebuilt=assemble_csv(paths,{s:100 for s in paths},csv)
        assert rebuilt['rows']==csv_info['rows'] and rebuilt['movie_rows']==csv_info['movie_rows']
        assert sha(csv)==sha(outputs/'submission.csv')==sha(outputs/'trajectory-complete/submission.csv')==csv_info['sha256']
        risk=runtime_evidence(proof['runtime']['movie_seconds'],times,build['runtime_projection_hours'],terminal['elapsed_seconds'])
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        actual=subprocess.run([str(BASE),str(ROOT/'scripts/get-kaggle-kernel-state.py'),'--kernel-slug',KERNEL],capture_output=True,text=True,env=env,check=True,timeout=45)
        state=json.loads(actual.stdout);assert state['present'] and state['current_version_number']==1
        result.update(status='public_production_verified_not_submittable_version',kernel=KERNEL,kernel_version=1,
            csv_sha256=csv_info['sha256'],rows=csv_info['rows'],runtime_risk=risk,public_movie_seconds=times,
            public_wall_seconds=terminal['elapsed_seconds'],ground_truth_opened=False,
            source_sha256=sha(Path(__file__)),remote_source_sha256=sha(remote/observed['code_file']),
            strict_quality_failures_preserved=True,competition_submission_authorized=False)
        persist();print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    except BaseException as error:
        result.update(status='verification_failed',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
