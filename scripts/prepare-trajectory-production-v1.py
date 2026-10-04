"""Stage production only after actual two-T4 eight-movie quality acceptance."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.submission_sharding import validate_submission_kernel_metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--acceptance-sha256',required=True)
    parser.add_argument('--overlap',action='store_true')
    args=parser.parse_args()
    proof=(ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-result.json' if args.overlap
           else ROOT/'reports/experiments/trajectory-kaggle-acceptance-v2-result.json')
    if sha(proof)!=args.acceptance_sha256:raise ValueError('Acceptance decision changed')
    acceptance=json.loads(proof.read_text())
    if (acceptance['status']!='acceptance_passed' or not acceptance['production_notebook_eligible']
            or not acceptance['runtime']['two_t4_verified']
            or not acceptance['comparison']['diagnostic_gate_passed']):
        raise ValueError('Actual T4 functionality and paired quality gates required')
    projection=max(acceptance['runtime'][('projected_199_movies_two_gpus_hours' if args.overlap
                                          else 'projected_199_movies_two_workers_hours')],
                   acceptance['runtime']['wall_seconds']/8*199/3600)
    # Policy revision2, explicitly logged before production: use measured full
    # cohort makespan (including startup/merge), at least TWO HOURS spare inside
    # the unchanged ten-hour stop, and two further hours below Kaggle's limit.
    # The previous50% multiplier was a provisional engineering heuristic, not a
    # competition/quality rule. Hidden embryo workload remains an explicit risk.
    if not 0<projection<=8:
        raise ValueError('Measured full-movie throughput lacks two hours of runtime headroom')
    build_path=(ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-build.json' if args.overlap
                else ROOT/'reports/experiments/trajectory-kaggle-runtime-v2-build.json')
    build=json.loads(build_path.read_text())
    if build['contract_sha256']!=acceptance['contract_sha256']:
        raise ValueError('Accepted runtime differs from deployment bundle')
    out=ROOT/('.biohub/staging/biohub-trajectory-overlap-candidate-v1' if args.overlap
              else '.biohub/staging/biohub-trajectory-motion-candidate-v1')
    out.mkdir(exist_ok=False)
    prior=ROOT/('.biohub/staging/biohub-trajectory-overlap-acceptance-v1' if args.overlap
                else '.biohub/staging/biohub-trajectory-motion-acceptance-v3')
    accepted_name='trajectory-overlap-acceptance.ipynb' if args.overlap else 'trajectory-motion-acceptance.ipynb'
    if args.overlap and sha(prior/accepted_name)!=build['notebook_sha256']:
        raise ValueError('Accepted overlay notebook changed')
    accepted=json.loads((prior/accepted_name).read_text())
    accepted_code=''.join(accepted['cells'][1]['source'])
    if args.overlap:
        acceptance_replay=accepted_code
    else:
        bootstrap=(ROOT/'scripts/trajectory-kaggle-bootstrap-v1.py').read_text()
        acceptance_replay=bootstrap.replace('__ARCHIVE_SHA256__',build['archive_sha256']).replace('__CONTRACT_SHA256__',build['contract_sha256']).replace('__RUN_MODE__','acceptance')
    if accepted_code!=acceptance_replay:raise ValueError('Bootstrap changed after actual acceptance')
    code=acceptance_replay.replace("RUN_MODE = 'acceptance'","RUN_MODE = 'production'",1)
    ast.parse(code)
    notebook=dict(accepted)
    notebook['cells']=[dict(cell_type='markdown',metadata={},id='candidate-description',source=[
        'Biohub source-trained trajectory motion candidate. Public LF-DCTTA backbone plus our fixed motion-expert endpoint repair. Offline two-GPU whole-movie inference; all input movies required. See runtime NOTICE and PROVENANCE.']),
        dict(cell_type='code',execution_count=None,metadata={},id='production-inference',outputs=[],source=code.splitlines(keepends=True))]
    filename='trajectory-motion-candidate.ipynb'
    (out/filename).write_text(json.dumps(notebook,indent=2))
    metadata=json.loads((prior/'kernel-metadata.json').read_text())
    metadata.update(id=('indarkarhana/biohub-trajectory-overlap-candidate' if args.overlap
                        else 'indarkarhana/biohub-trajectory-motion-candidate'),
                    title='Biohub Trajectory Motion Candidate',code_file=filename)
    validate_submission_kernel_metadata(metadata)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    receipt=dict(status='production_staged_not_launched',kernel=metadata['id'],
                 acceptance_sha256=args.acceptance_sha256,contract_sha256=build['contract_sha256'],
                 notebook_sha256=sha(out/filename),runtime_projection_hours=projection,
                 runtime_budget_policy_revision=2,minimum_runtime_headroom_hours=2,
                 observed_runtime_headroom_hours=10-projection,inference_watchdog_seconds=36000,
                 notebook_wall_cap_seconds=43200,public_test_run_required=True,
                 candidate_is_not_exact_public_replica=True,submission_performed=False)
    receipt_path=ROOT/('reports/experiments/trajectory-overlap-production-v1-build.json' if args.overlap
                      else 'reports/experiments/trajectory-production-v1-build.json')
    receipt_path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
