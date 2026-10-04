"""Prove overlap preserves all eight FP32 graphs; reuse frozen quality evidence."""
import json
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,verify_bundle,validate_graph


def pinned(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen evidence changed: '+path.name)
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exact-tta-cache',action='store_true')
    parser.add_argument('--terminal-sha256');parser.add_argument('--manifest-sha256')
    args=parser.parse_args()
    run_id='trajectory-overlap-cache-v1' if args.exact_tta_cache else 'trajectory-overlap-v1'
    if args.exact_tta_cache and not (args.terminal_sha256 and args.manifest_sha256):
        raise ValueError('Frozen cache experiment terminal and transport hashes required')
    target=ROOT/f'reports/experiments/{run_id}-result.json'
    if target.exists():raise ValueError('Preserve completed decision')
    folder=ROOT/('.biohub/cache/trajectory-overlap-cache-full-v1-output' if args.exact_tta_cache
                 else '.biohub/cache/trajectory-overlap-full-v1-output')
    terminal=pinned(folder/'result.json',args.terminal_sha256 or 'acccae6c68fea1044689a04e0774fc0518c9dc9c698bc824eeb375f61971c4cb')
    baseline=ROOT/'.biohub/cache/trajectory-division-full-v1-output'
    prior=pinned(baseline/'result.json','40c35432a298d22f347cbdd111b1cd6d21ea3965acc35b4fc63369d3147544b1')
    contract=('1b0124fd435802c47fffd072ed323d4bfde3aa95785db05d7a7eb9610fbdb7d1' if args.exact_tta_cache
              else 'cb3a8e74a63f69ac270ea5c6995c7260dcd80ea15dcbd1faefa95586681cef1c')
    verify_bundle(ROOT/f'.biohub/cache/{run_id}-bundle',contract)
    if (terminal['status']!='complete_prelabel_predictions' or terminal['mode']!='full'
            or terminal['contract_sha256']!=contract or not terminal['inputs_unchanged']
            or terminal['ground_truth_opened'] or set(terminal['movies'])!=set(prior['movies'])
            or terminal['concurrent_workers']!=2 or terminal['gpu_count']!=1):
        raise ValueError('Wrong or incomplete overlap run')
    inventory=pinned(folder/'REMOTE_ARTIFACT_MANIFEST.json',
                     args.manifest_sha256 or '63dea5332b1b26752e9e4b51a8484f078f7c77e972a05bc50d63566ee904376f')
    for row in inventory['files']:
        path=folder/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Downloaded artifact changed')
    records=[]
    for stem,arms in terminal['movies'].items():
        if arms['original']['frames']!=100:raise ValueError('Full movies required')
        for name,field in (('prediction.json','prediction_sha256'),
                           ('repaired-prediction.json','repaired_sha256')):
            old=baseline/(stem+'-original')/name
            new=folder/('shard-'+str(arms['shard']))/(stem+'-original')/name
            old_graph=pinned(old,prior['movies'][stem]['original'][field])
            new_graph=pinned(new,arms['original'][field]);validate_graph(new_graph,100)
            if new_graph!=old_graph:raise ValueError('Changed predictions require a new quality evaluation')
            records.append(dict(movie=stem,filename=name,exact_graph_identity=True,sha256=sha(new)))
    quality=pinned(ROOT/'reports/experiments/trajectory-division-full-movie-v1-result.json',
                   '6dc733376a89e0eb8fdb8b996caf008252c84799199d35ef5f7f48a6ff39bb40')
    if quality['status']!='diagnostic_pass':raise ValueError('Baseline quality missing')
    result=dict(status='exact_graph_identity_passed',run_id=run_id,contract_sha256=contract,
                terminal_sha256=sha(folder/'result.json'),records=records,
                ground_truth_opened=False,score_reused_without_reopening_truth=True,
                baseline_seconds=prior['elapsed_seconds'],overlap_seconds=terminal['elapsed_seconds'],
                walltime_reduction_fraction=1-terminal['elapsed_seconds']/prior['elapsed_seconds'],
                speedup=prior['elapsed_seconds']/terminal['elapsed_seconds'],
                two_t4_acceptance_required=True,authorized_for_submission=False,
                total_cpu_threads=2,aggregate_cuda_allocator_fraction=.70)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
