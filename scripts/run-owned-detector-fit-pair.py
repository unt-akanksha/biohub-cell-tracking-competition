"""Sequential paired detector fits within one launcher watchdog/process group."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,'-u',str(args.runtime/'run_pilot.py'),'--repo',str(args.repo),'--runtime',str(args.runtime),
        '--checkpoint',str(args.checkpoint),'--manifest',str(args.manifest),'--data',str(args.data),
        '--logit-targets','--fit-profile','full']
    results={}
    for arm in ('sparse','pu'):
        subprocess.run(common+['--objective',arm,'--output',str(args.output/arm)],check=True)
        results[arm]=json.loads((args.output/arm/'result.json').read_text())
        (args.output/'pair_progress.json').write_text(json.dumps(dict(completed_arms=list(results),authorized_for_submission=False)))
    if (results['sparse']['input_hashes']!=results['pu']['input_hashes']
        or results['sparse']['target_hashes']!=results['pu']['target_hashes']):
        raise ValueError('Paired objectives did not consume identical source images/teacher targets')
    result=dict(status='completed_detector_pair_not_selection',paired_inputs_identical=True,paired_targets_identical=True,
        checkpoints={arm:r['checkpoint_sha256'] for arm,r in results.items()},steps_per_arm=1000,
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    (args.output/'pair_result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
