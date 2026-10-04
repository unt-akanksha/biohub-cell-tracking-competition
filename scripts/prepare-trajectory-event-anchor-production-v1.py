"""Stage the accepted fixed runtime for a public-test production run, not submit."""
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.submission_sharding import validate_submission_kernel_metadata


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    proof_path=ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-result.json';proof=read(proof_path)
    assert proof['status']=='acceptance_passed' and proof['production_notebook_eligible']
    assert proof['runtime']['two_t4_verified'] and proof['runtime']['headroom_passed']
    assert proof['exact_graph_quality_reused'] and len(proof['records'])==8
    assert proof['quality_tradeoff_release_review_required'] and proof['selection_failure_preserved'] and proof['validation_movie_failure_preserved']
    build=read(ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-build.json')
    prior=ROOT/'.biohub/staging/biohub-event-anchor-acceptance-v1';metadata=read(prior/'kernel-metadata.json')
    assert proof['contract_sha256']==build['contract_sha256']
    assert sha(prior/metadata['code_file'])==build['notebook_sha256']
    assert sha(prior/'overlays.json')==build['overlay_sha256']
    assert sha(prior/'derived-contract.json')==build['contract_sha256']
    accepted=read(prior/metadata['code_file']);code=''.join(accepted['cells'][1]['source'])
    marker="RUN_MODE = 'acceptance'";assert code.count(marker)==1
    production=code.replace(marker,"RUN_MODE = 'production'");ast.parse(production)
    assert production.replace("RUN_MODE = 'production'",marker)==code
    filename='event-anchor-candidate.ipynb'
    metadata.update(id='indarkarhana/biohub-event-anchor-candidate',title='Biohub Event Anchor Candidate',code_file=filename)
    validate_submission_kernel_metadata(metadata)
    notebook=dict(accepted,cells=[dict(cell_type='markdown',metadata={},source=[
        'Public-test production run of our fixed event-assignment candidate. Exact accepted offline two-T4 runtime, four workers, no model or threshold changes. No selective gate is included. The embedded bundle contract and NOTICE retain their acceptance-time provenance flags; the separate production/release receipts determine eligibility. Known selection and validation regressions remain disclosed. This run produces submission.csv but does not enter the competition.']),
        dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=production.splitlines(keepends=True))])
    target=ROOT/'.biohub/staging/biohub-event-anchor-candidate-v1';target.mkdir(exist_ok=False)
    (target/filename).write_bytes(json.dumps(notebook,indent=2).encode())
    (target/'kernel-metadata.json').write_bytes(json.dumps(metadata,indent=2).encode())
    for name in ('overlays.json','derived-contract.json'):(target/name).write_bytes((prior/name).read_bytes())
    result=dict(status='public_production_test_staged_not_launched',kernel=metadata['id'],
        acceptance_sha256=sha(proof_path),contract_sha256=build['contract_sha256'],overlay_sha256=build['overlay_sha256'],
        notebook_sha256=sha(target/filename),event_portable_proof_sha256=build['event_portable_proof_sha256'],
        required_gpus=2,maximum_workers=4,acceptance_wall_cap_seconds=3600,
        runtime_projection_hours=proof['runtime']['projected_199_movies_two_gpus_hours'],
        only_executable_change_from_acceptance_is_run_mode=True,model_or_threshold_changed=False,
        contract_flags_are_acceptance_time_provenance=True,quality_failures_preserved=True,
        quality_tradeoff_release_review_required=True,competition_submission_authorized=False,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-event-anchor-production-v1-build.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
