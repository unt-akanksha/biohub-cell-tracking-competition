"""Verify actual two-T4 smoke, eight exact graphs, CSV and runtime headroom."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph,assemble_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs',type=Path,required=True);args=p.parse_args()
    started=time.monotonic();report=ROOT/'reports/experiments/trajectory-structured-kaggle-v1-result.json'
    assert not report.exists()
    build=json.loads((ROOT/'reports/experiments/trajectory-structured-kaggle-v1-build.json').read_text())
    contract=build['contract_sha256']
    assert contract=='4e13ec7dea134c2da7b13133d8ed5ba92d5bb392954f35e26c29288c446a4695'
    quality_path=ROOT/'reports/experiments/trajectory-structured-eight-v1-result.json'
    assert sha(quality_path)=='0469b99ccf3d347a72ec00afb7206ea8fa6bd6498a7c388e09b0d8cf667508ce'
    quality=json.loads(quality_path.read_text());assert quality['quality_pass'] and all(quality['gates'].values())
    reference=ROOT/'.biohub/cache/trajectory-structured-eight-v1'
    accepted=json.loads((ROOT/'reports/experiments/trajectory-overlap-cache-kaggle-v2-result.json').read_text())
    reference_root=ROOT/'.biohub/cache/trajectory-overlap-cache-kaggle-v1-output'
    prior={r['movie']:reference_root/r['path'] for r in accepted['verified_graphs'] if r['arm']=='repaired'}
    terminal_path=args.outputs/'trajectory-complete/result.json';terminal=json.loads(terminal_path.read_text())
    smoke=json.loads((args.outputs/'trajectory-smoke/result.json').read_text())
    for result,mode in ((terminal,'validation'),(smoke,'smoke')):
        assert result['status']=='complete' and result['mode']==mode
        assert result['contract_sha256']==contract and result['gpu_count']==2 and result['worker_count']==4
        assert not result['ground_truth_opened'] and not result['submission_created']
        assert set(result['workers'])=={'0','1','2','3'}
        for worker in result['workers'].values():
            assert worker['status'] in ('functionality_passed','complete_prelabel_predictions') and worker['inputs_unchanged']
            assert 'T4' in worker['device'] and worker['solver_backend']=='SCIP'
            assert worker['contract_sha256']==contract and not worker['ground_truth_opened']
            assert all(r['original']['frames']==(8 if mode=='smoke' else 100) for r in worker['movies'].values())
    assert 0<terminal['elapsed_seconds']<=2600
    paths={};records=[];movie_seconds=[]
    for shard,worker in terminal['workers'].items():
        for stem,arms in worker['movies'].items():
            assert stem in quality['records'] and stem not in paths and set(arms)=={'original'}
            folder=args.outputs/'trajectory-complete'/('shard-'+shard)/(stem+'-original')
            path=folder/'repaired-prediction.json';row=arms['original']
            assert sha(path)==row['repaired_sha256']
            graph=json.loads(path.read_text());validate_graph(graph,100)
            ref=reference/(stem+'-prediction.json')
            assert sha(ref)==quality['records'][stem]['prediction_sha256']
            assert graph==json.loads(ref.read_text()), 'Runtime graph differs from frozen quality graph: '+stem
            baseline=folder/'motion-repaired-prediction.json'
            assert sha(prior[stem])==quality['records'][stem]['baseline_sha256']
            assert json.loads(baseline.read_text())==json.loads(prior[stem].read_text())
            details=json.loads((folder/'repair-details.json').read_text())['structured_assignment']
            assert details['node_positions_unchanged'] and details['degrees_unchanged'] and details['division_and_gap_incident_edges_unchanged']
            paths[stem]=path;movie_seconds.append(row['seconds'])
            records.append(dict(stem=stem,path=str(path),sha256=sha(path),exact_frozen_graph_identity=True))
    assert set(paths)==set(quality['records']) and len(paths)==8
    verification=ROOT/'.biohub/cache/trajectory-structured-kaggle-v1-verification';verification.mkdir(exist_ok=False)
    csv_path=args.outputs/'trajectory-complete/validation-predictions.csv'
    rebuilt=verification/'validation-predictions.csv';assemble_csv(paths,{s:100 for s in paths},rebuilt)
    assert sha(csv_path)==sha(rebuilt)
    projection=terminal['elapsed_seconds']/8*199/3600
    assert projection<=8., 'Insufficient measured headroom for ten-hour production watchdog'
    result=dict(status='acceptance_passed',contract_sha256=contract,terminal_sha256=sha(terminal_path),
                quality_sha256=sha(quality_path),records=records,source_prefilter_movie_regression_preserved=True,
                ground_truth_opened=False,exact_graph_quality_reused=True,summaries=quality['summaries'],
                rows=quality['rows'],by_embryo=quality['by_embryo'],per_movie_summaries=quality['per_movie'],
                comparison=dict(diagnostic_gate_passed=True,source_prefilter_movie_regression_preserved=True),
                runtime=dict(two_t4_verified=True,worker_count=4,wall_seconds=terminal['elapsed_seconds'],
                             movie_seconds=movie_seconds,projected_199_movies_two_gpus_hours=projection,
                             projection_is_not_hidden_runtime_guarantee=True),
                production_notebook_eligible=True,submission_performed=False,elapsed_seconds=time.monotonic()-started)
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','rows','by_embryo','per_movie_summaries')}))


if __name__=='__main__':main()
