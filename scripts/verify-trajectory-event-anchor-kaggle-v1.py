"""Verify actual offline T4 stages, exact validation graphs, CSV and throughput."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph,assemble_csv
from research.trajectory_event_release_v1 import check_event_stage


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outputs',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();report=ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-result.json'
    assert not report.exists()
    build=read(ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-build.json')
    contract=build['contract_sha256'];assert contract=='e8f1c79e6309fb9458c64e9e879b2576d0d570d61856cffafcb64cb50b6a6801'
    quality_path=ROOT/'reports/experiments/trajectory-event-anchor-eight-v1.json'
    assert sha(quality_path)=='f56568612879bf5d7f604a459b17ca2a7f4f6185920f51af8e0cdccfa3d465b1'
    quality=read(quality_path);assert quality['selection_failure_preserved'] and not quality['overall_release_gates_pass']
    reference=ROOT/'.biohub/cache/trajectory-event-anchor-eight-v1'
    prior=ROOT/'.biohub/cache/trajectory-event-eight-v1-features';inputs=read(prior/'RESULT.json')
    assert sha(prior/'RESULT.json')==quality['input_receipt_sha256']
    terminal_path=args.outputs/'trajectory-complete/result.json';terminal=read(terminal_path)
    smoke=read(args.outputs/'trajectory-smoke/result.json')
    for result,mode in ((terminal,'validation'),(smoke,'smoke')):
        assert result['status']=='complete' and result['mode']==mode
        assert result['contract_sha256']==contract and result['gpu_count']==2 and result['worker_count']==4
        assert not result['ground_truth_opened'] and not result['submission_created']
        assert set(result['workers'])=={'0','1','2','3'}
        for shard,worker in result['workers'].items():
            assert worker['status'] in ('functionality_passed','complete_prelabel_predictions') and worker['inputs_unchanged']
            assert 'T4' in worker['device'] and worker['solver_backend']=='SCIP'
            assert worker['contract_sha256']==contract and not worker['ground_truth_opened']
            for stem,arms in worker['movies'].items():
                assert set(arms)=={'original'} and arms['original']['frames']==(8 if mode=='smoke' else 100)
                folder=args.outputs/('trajectory-smoke' if mode=='smoke' else 'trajectory-complete')/('shard-'+shard)/(stem+'-original')
                details=read(folder/'repair-details.json')['event_assignment']
                check_event_stage(read(folder/'pre-postprocess.json'),read(folder/'structured-repaired-prediction.json'),
                    read(folder/'repaired-prediction.json'),details,frames=8 if mode=='smoke' else 100)
    assert 0<terminal['elapsed_seconds']<=3600
    paths={};records=[];movie_seconds=[];event_seconds=[]
    for shard,worker in terminal['workers'].items():
        for stem,arms in worker['movies'].items():
            assert stem in quality['inference'] and stem not in paths
            folder=args.outputs/'trajectory-complete'/('shard-'+shard)/(stem+'-original')
            path=folder/'repaired-prediction.json';row=arms['original']
            assert sha(path)==row['repaired_sha256']
            graph=read(path);validate_graph(graph,100)
            ref=reference/(stem+'-prediction.json');assert sha(ref)==quality['inference'][stem]['prediction_sha256']
            assert graph==read(ref),'Frozen quality graph differs on actual T4 run: '+stem
            baseline=folder/'structured-repaired-prediction.json';bp=prior/(stem+'-prediction.json')
            assert sha(bp)==inputs['per_movie'][stem]['prediction_sha256'] and read(baseline)==read(bp)
            ip=folder/'pre-postprocess.json';assert read(ip)==read(ROOT/inputs['per_movie'][stem]['initial_path'])
            details=read(folder/'repair-details.json');structured=details['structured_assignment'];event=details['event_assignment']
            assert structured['node_positions_unchanged'] and structured['degrees_unchanged'] and structured['division_and_gap_incident_edges_unchanged']
            assert not event['budget_exhausted'] and not event['solver_fallbacks'],'Event stage did not complete its editable scope'
            paths[stem]=path;movie_seconds.append(row['seconds']);event_seconds.append(event['seconds'])
            records.append(dict(stem=stem,path=str(path),sha256=sha(path),exact_frozen_graph_identity=True,
                baseline_stage_exact=True,added_edges=event['added_edges'],removed_edges=event['removed_edges']))
    assert len(paths)==8 and set(paths)==set(quality['inference'])
    verification=ROOT/'.biohub/cache/trajectory-event-anchor-kaggle-v1-verification';verification.mkdir(exist_ok=False)
    rebuilt=verification/'validation-predictions.csv';assemble_csv(paths,{s:100 for s in paths},rebuilt)
    csv_path=args.outputs/'trajectory-complete/validation-predictions.csv';assert sha(csv_path)==sha(rebuilt)
    projection=terminal['elapsed_seconds']/8*199/3600
    runtime_pass=projection<=8.
    result=dict(status='acceptance_passed' if runtime_pass else 'runtime_headroom_failed',
        contract_sha256=contract,terminal_sha256=sha(terminal_path),quality_sha256=sha(quality_path),records=records,
        ground_truth_opened=False,exact_graph_quality_reused=True,summaries=quality['summaries'],rows=quality['rows'],
        by_embryo=quality['by_embryo'],per_movie_summaries=quality['per_movie_summaries'],
        quality_checks=quality['quality_checks'],selection_failure_preserved=True,validation_movie_failure_preserved=True,
        runtime=dict(two_t4_verified=True,worker_count=4,wall_seconds=terminal['elapsed_seconds'],
            movie_seconds=movie_seconds,event_seconds=event_seconds,projected_199_movies_two_gpus_hours=projection,
            projection_is_not_hidden_runtime_guarantee=True,headroom_passed=runtime_pass),
        production_notebook_eligible=runtime_pass,quality_tradeoff_release_review_required=True,
        submission_performed=False,source_sha256=sha(Path(__file__)),release_helper_sha256=sha(ROOT/'research/trajectory_event_release_v1.py'),
        elapsed_seconds=time.monotonic()-started)
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','rows','by_embryo','per_movie_summaries')}),flush=True)


if __name__=='__main__':main()
