"""Verify the completed exact production kernel, then optionally submit once."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph,assemble_csv
from research.submission_sharding import validate_submission_kernel_metadata

KERNEL='indarkarhana/biohub-trajectory-motion-candidate'
COMPETITION='biohub-cell-tracking-during-development'
CONTRACT='d10d19b1e21b30e2eb5c6bf1c05161559f74594ce30e250e3c473f2d8b0379f2'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=Path,required=True)
    p.add_argument('--terminal-sha256',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    receipt=ROOT/'reports/experiments/trajectory-motion-v1-submission.json'
    if receipt.exists():raise ValueError('Submission already attempted; inspect actual state before retry')
    build=json.loads((ROOT/'reports/experiments/trajectory-production-v1-build.json').read_text())
    launch=json.loads((ROOT/'reports/experiments/trajectory-production-v1-launch.json').read_text())
    quality=ROOT/'reports/experiments/trajectory-kaggle-acceptance-v2-result.json'
    if (sha(quality)!=build['acceptance_sha256'] or build['contract_sha256']!=CONTRACT
            or launch['notebook_sha256']!=build['notebook_sha256']
            or not json.loads(quality.read_text())['comparison']['diagnostic_gate_passed']):
        raise ValueError('Accepted candidate provenance changed')
    staging=ROOT/'.biohub/staging/biohub-trajectory-motion-candidate-v1'
    metadata=json.loads((staging/'kernel-metadata.json').read_text())
    validate_submission_kernel_metadata(metadata)
    if metadata['id']!=KERNEL or sha(staging/metadata['code_file'])!=build['notebook_sha256']:
        raise ValueError('Frozen notebook changed')
    folder=args.outputs/'trajectory-complete'
    path=folder/'result.json'
    if sha(path)!=args.terminal_sha256:raise ValueError('Downloaded terminal changed')
    terminal=json.loads(path.read_text())
    if (terminal['status']!='complete' or terminal['mode']!='production'
            or terminal['gpu_count']!=2 or not terminal['submission_created']
            or terminal['ground_truth_opened'] or terminal['contract_sha256']!=CONTRACT
            or set(terminal['workers'])!={'0','1'} or not 0<terminal['elapsed_seconds']<36000):
        raise ValueError('Complete exact two-GPU production run required')
    smoke=json.loads((args.outputs/'trajectory-smoke/result.json').read_text())
    if (smoke['status']!='complete' or smoke['mode']!='smoke'
            or smoke['gpu_count']!=2 or smoke['contract_sha256']!=CONTRACT):
        raise ValueError('Same-contract two-GPU smoke missing')
    plans={str(s['shard_index']):s for s in terminal['shards']}
    outputs={};frames={};added=0;public_movie_seconds=[]
    for shard,worker in terminal['workers'].items():
        if (worker['status']!='complete_prelabel_predictions' or worker['contract_sha256']!=CONTRACT
                or not worker['inputs_unchanged'] or worker['ground_truth_opened']
                or 'T4' not in worker['device'] or worker['solver_backend']!='SCIP'
                or set(worker['movies'])!=set(plans[shard]['movie_ids'])):
            raise ValueError('Incomplete or wrong worker')
        for movie,arms in worker['movies'].items():
            if movie in outputs or set(arms)!={'original'}:raise ValueError('Duplicate/wrong movie')
            row=arms['original'];frames[movie]=row['frames'];added+=row['added_edges']
            public_movie_seconds.append(row['seconds'])
            base_path=folder/f'shard-{shard}/{movie}-original/prediction.json'
            new_path=folder/f'shard-{shard}/{movie}-original/repaired-prediction.json'
            if sha(base_path)!=row['prediction_sha256'] or sha(new_path)!=row['repaired_sha256']:
                raise ValueError('Frozen prediction changed')
            base=json.loads(base_path.read_text());new=json.loads(new_path.read_text())
            validate_graph(base,row['frames']);validate_graph(new,row['frames'])
            old_edges={(e['source_id'],e['target_id']) for e in base['edges']}
            new_edges={(e['source_id'],e['target_id']) for e in new['edges']}
            if (base['nodes']!=new['nodes'] or not old_edges<=new_edges
                    or len(new_edges-old_edges)!=row['added_edges']):
                raise ValueError('Repair violated original graph preservation')
            outputs[movie]=new_path
    if set(outputs)!=set(terminal['csv']['movie_rows']):raise ValueError('CSV movie coverage mismatch')
    # Independently enumerated from all public-test API pages at01:50UTC.
    # This guard applies to this local public-run verifier, NEVER hidden inference.
    if set(outputs)!={'44b6_0113de3b','44b6_0b24845f','6bba_05b6850b','6bba_05db0fb1'}:
        raise ValueError('Public-run output must cover all four independently listed test movies')
    csv_hash=terminal['csv']['sha256']
    if sha(folder/'submission.csv')!=csv_hash or sha(args.outputs/'submission.csv')!=csv_hash:
        raise ValueError('Root and assembled submission hashes differ')
    replay=args.outputs/'independent-submission-replay.csv'
    if not replay.exists():
        regenerated=assemble_csv(outputs,frames,replay)
        if regenerated['rows']!=terminal['csv']['rows']:raise ValueError('CSV row count changed')
    if sha(replay)!=csv_hash:raise ValueError('Independent CSV reconstruction differs')
    # Include the newly observed dense public movie; do not repeat the earlier
    # validation-only two-hour-buffer claim after seeing newer timing evidence.
    prior_times=json.loads(quality.read_text())['runtime']['movie_seconds']
    public_mean_projection=sum(public_movie_seconds)/4*199/2/3600
    combined_projection=(sum(prior_times)+sum(public_movie_seconds))/12*199/2/3600
    if max(public_mean_projection,combined_projection)>=10:
        raise ValueError('Observed mean workload projects beyond the unchanged inference cutoff')
    runtime_risk=dict(validation_cohort_projection_hours=build['runtime_projection_hours'],
                      public_mean_movie_projection_hours=public_mean_projection,
                      combined_twelve_movie_projection_hours=combined_projection,
                      public_four_movie_makespan_projection_hours=terminal['elapsed_seconds']/4*199/3600,
                      hidden_inference_cutoff_hours=10,
                      two_hour_planning_buffer_not_confirmed_by_new_public_timing=True,
                      dense_movie_and_static_shard_imbalance_timeout_risk=True,
                      user_informed_before_submission=True,hidden_runtime_guaranteed=False)
    cli=str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    def run(command,timeout=60):
        return subprocess.run(command,capture_output=True,text=True,encoding='utf-8',
                              check=True,timeout=timeout,env=env)
    state=json.loads(run([sys.executable,str(ROOT/'scripts/get-kaggle-kernel-state.py'),
                          '--kernel-slug',KERNEL]).stdout)
    if (not state['present'] or state['current_version_number']!=1 or not state['is_private']
            or not state['enable_gpu'] or state['enable_tpu'] or state['enable_internet']
            or state['competition_sources']!=metadata['competition_sources']
            or state['dataset_sources']!=metadata['dataset_sources']
            or state['kernel_sources'] or state['model_sources']):
        raise ValueError('Exact immutable remote version/inputs/settings required')
    if 'KernelWorkerStatus.COMPLETE' not in run([cli,'kernels','status',KERNEL]).stdout:
        raise ValueError('Production notebook is not complete')
    result=dict(status='verified_not_submitted',utc=datetime.now(timezone.utc).isoformat(),
                kernel=KERNEL,kernel_version=1,contract_sha256=CONTRACT,
                notebook_sha256=build['notebook_sha256'],terminal_sha256=args.terminal_sha256,
                acceptance_sha256=sha(quality),csv_sha256=csv_hash,rows=terminal['csv']['rows'],
                movies=sorted(outputs),added_edges=added,frames=frames,
                official_validation_score=.9453907265031998,public_leaderboard_score=None,
                independently_held_out=False,public_backbone_training_overlap=True,
                public_leaderboard_used_for_selection=False,submission_performed=False,
                remote_state=state,production_wall_seconds=terminal['elapsed_seconds'],
                runtime_risk=runtime_risk,verifier_sha256=sha(Path(__file__)))
    if not args.execute:
        print(json.dumps(result,indent=2));return
    quota=json.loads(run([cli,'quota','--format','json']).stdout)
    gpu=[r for r in quota if r['resource']=='GPU']
    if len(gpu)!=1 or float(gpu[0]['remaining'].removesuffix('h'))-20<8:
        raise ValueError('Conservative hidden-run reservation violates eight-hour reserve')
    raw=run([cli,'competitions','submissions',COMPETITION,'--page-size','5','--format','json']).stdout
    rows=json.loads(raw[raw.index('['):])
    dates=[datetime.fromisoformat(r['date'].replace('Z','+00:00')).replace(tzinfo=timezone.utc) for r in rows]
    if dates!=sorted(dates,reverse=True):raise ValueError('Submission history order ambiguous')
    today=datetime.now(timezone.utc).date();daily=sum(d.date()==today for d in dates)
    if daily>=5:raise ValueError('Daily submission limit reached')
    result.update(status='submission_requested',quota=quota,worst_case_reserved_gpu_hours=20,
                  daily_submissions_before=daily,reserve_hours=8)
    receipt.write_text(json.dumps(result,indent=2))
    try:
        response=run([cli,'competitions','submit',COMPETITION,'-k',KERNEL,'-v','1','-f','submission.csv',
                      '-m','Clean LF-DCTTA plus source-trained trajectory endpoint repair; exact 8-movie paired gate; offline two-T4'],timeout=120)
        result.update(status='submitted',submission_performed=True,kaggle_response=response.stdout)
    except BaseException as error:
        result.update(status='submission_outcome_requires_inspection',error=type(error).__name__)
        raise
    finally:
        receipt.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
