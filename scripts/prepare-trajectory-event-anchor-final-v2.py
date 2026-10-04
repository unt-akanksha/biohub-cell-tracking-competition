"""Record a disclosed exploratory release decision and stage unchanged inference.

This does not waive or overwrite the earlier strict non-regression gates.
It does not submit or select a final entry.
"""
import ast
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.submission_sharding import validate_submission_kernel_metadata


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    reports = ROOT / 'reports/experiments'
    receipt = reports / 'trajectory-event-anchor-final-v2-build.json'
    target = ROOT / '.biohub/staging/biohub-event-anchor-candidate-v2'
    assert not receipt.exists() and not target.exists()
    proof_path = reports / 'trajectory-event-anchor-production-v1-verification-r2.json'
    proof = read(proof_path)
    assert proof['status'] == 'public_production_verified_not_submittable_version'
    assert len(proof['records']) == 8 and all(r['three_stages_exact'] for r in proof['records'])
    assert not proof['ground_truth_opened'] and proof['strict_quality_failures_preserved']
    previous = read(reports / 'trajectory-event-anchor-production-v1-build.json')
    stage = ROOT / '.biohub/staging/biohub-event-anchor-candidate-v1'
    metadata = read(stage / 'kernel-metadata.json')
    validate_submission_kernel_metadata(metadata)
    assert sha(stage / metadata['code_file']) == previous['notebook_sha256']
    assert sha(stage / 'derived-contract.json') == previous['contract_sha256']
    assert sha(stage / 'overlays.json') == previous['overlay_sha256']
    cohorts = {}
    for name, filename, arm, count in (
        ('source8', 'trajectory-event-anchor-source-v1', 'event_smoke', 8),
        ('source52', 'trajectory-event-anchor-expanded-source-v1', 'anchor', 52),
        ('selection10', 'trajectory-event-anchor-selection-v1', 'anchor', 10),
        ('validation8', 'trajectory-event-anchor-eight-v1', 'anchor', 8),
    ):
        path = reports / (filename + '.json')
        data = read(path)
        assert data['model_sha256'] == '508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
        old, new = data['summaries']['baseline'], data['summaries'][arm]
        assert old['n'] == new['n'] == count
        assert new['score'] > old['score'] and new['edge_jaccard'] >= old['edge_jaccard']
        assert all(old[k] == new[k] for k in ('division_tp', 'division_fp', 'division_fn'))
        movies = data['per_movie_summaries']
        regressions = [dict(stem=s, delta=movies[arm][s]['score'] - value['score'])
                       for s, value in movies['baseline'].items()
                       if movies[arm][s]['score'] < value['score'] - 1e-12]
        cohorts[name] = dict(sha256=sha(path), summaries=data['summaries'],
                             by_embryo=data['by_embryo'], regressions=regressions,
                             prior_quality_checks=data.get('quality_checks'))
    selection = read(reports / 'trajectory-event-anchor-selection-v1.json')
    validation = read(reports / 'trajectory-event-anchor-eight-v1.json')
    assert not selection['quality_checks_pass'] and not validation['overall_release_gates_pass']
    assert selection['source_role_disjoint'] and not selection['selection_used_for_training']
    assert validation['source_and_selection_role_disjoint'] and not validation['model_or_threshold_refitted']
    assert selection['all_predictions_frozen_before_scoring'] and validation['all_predictions_frozen_before_scoring']
    gate_path = reports / 'trajectory-correction-gate-oof-v1.json'
    gate = read(gate_path)
    assert not gate['eligible_for_separate_selection_test']
    accepted = read(stage / metadata['code_file'])
    code = ''.join(accepted['cells'][1]['source'])
    ast.parse(code)
    assert "RUN_MODE = 'production'" in code
    assert "else 35900" in code and "timeout=cap+60" in code
    notebook = dict(accepted)
    notebook['cells'] = [dict(cell_type='markdown', metadata={}, source=[
        'Experimental clean event-assignment candidate. Licensed public image ensemble plus our source-trained trajectory repair, structured assignment, and fixed event alternatives. Inference is byte-identical to the verified public test; no selector or movie-identity router is added. Complete-movie gains and individual/selection-embryo regressions are disclosed in the external release receipt. Earlier strict quality gates remain failed; this is not a universal non-regression claim or selected final entry. The launch requests a 12-hour platform cap; inference retains its existing sub-10-hour internal watchdog. Embedded contract flags preserve acceptance-time provenance.']),
        accepted['cells'][1]]
    assert ''.join(notebook['cells'][1]['source']) == code
    target.mkdir()
    for name in ('kernel-metadata.json', 'overlays.json', 'derived-contract.json'):
        (target / name).write_bytes((stage / name).read_bytes())
    (target / metadata['code_file']).write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    result = dict(
        status='exploratory_candidate_staged_requires_final_saved_run',
        utc=datetime.now(timezone.utc).isoformat(), kernel=metadata['id'], expected_kernel_version=2,
        notebook_sha256=sha(target / metadata['code_file']), contract_sha256=previous['contract_sha256'],
        overlay_sha256=previous['overlay_sha256'], public_verification_sha256=sha(proof_path),
        public_test_notebook_sha256=previous['notebook_sha256'],
        executable_cells_identical_to_verified_public_test=True, only_notebook_change_is_disclosure_markdown=True,
        requested_platform_timeout_seconds=43200, internal_inference_watchdog_seconds=36000,
        runtime_risk=proof['runtime_risk'], cohorts=cohorts,
        gate_experiment_sha256=sha(gate_path), failed_gate_included=False,
        strict_quality_gates_pass=False, strict_quality_failures_preserved=True,
        exploratory_entry_rationale='Consistent pooled and raw-edge gains in four separately reported cohorts; unchanged node locations and division counts. Accept recorded movie/selection-embryo risk for one experimental entry, not universal promotion.',
        prior_pipeline_and_public_backbone_exposure_disclosed=True,
        quality_decision='one_exploratory_entry_after_final_runtime_and_output_verification',
        user_runtime_and_quality_risks_disclosed=True,
        model_or_threshold_changed=False, metric_hack_or_movie_identity_routing=False,
        final_selection_authorized=False, submission_performed=False,
        source_sha256=sha(Path(__file__)),
    )
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'cohorts'}), flush=True)


if __name__ == '__main__':
    main()
