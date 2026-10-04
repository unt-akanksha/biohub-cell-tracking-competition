"""Repeat public verification with a diagnosed CPU-only replay budget revision.

The first failed receipt and the production notebook remain immutable.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'scripts/verify-trajectory-event-anchor-production-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == 'cf3f526d093de36233606c48d7e750e8e74271056df80b9213ecfa309274944c'
    previous = ROOT / 'reports/experiments/trajectory-event-anchor-production-v1-verification.json'
    diagnostic = ROOT / 'reports/experiments/trajectory-event-public-replay-budget-v1.json'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    failure, proof = read(previous), read(diagnostic)
    assert failure['status'] == 'verification_failed'
    assert 'Event replay differs: 6bba_05db0fb1' in failure['error']
    assert proof['status'] == 'local_replay_budget_diagnosed'
    assert not proof['production_candidate_changed'] and not proof['annotations_opened']
    normal, extended = proof['records']['default_cpu'], proof['records']['extended_cpu_diagnostic']
    assert normal['budget_exhausted'] and not normal['exact_kaggle_graph']
    for arm in (extended, proof['actual_kaggle_details']):
        assert arm['processed_frames'] == 99 and not arm['budget_exhausted'] and not arm['solver_fallbacks']
    assert extended['exact_kaggle_graph'] and extended['symmetric_edge_difference'] == 0
    provenance = dict(
        previous_failed_verification_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        cpu_budget_diagnosis_sha256=hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
        original_verifier_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        local_replay_per_frame_seconds=10, local_replay_stage_seconds=600,
        production_notebook_or_model_changed=False,
        actual_kaggle_default_budget_checks_unchanged=True,
    )
    source = base.read_text(encoding='utf-8')
    replacements = [
        ('trajectory-event-anchor-production-v1-verification.json', 'trajectory-event-anchor-production-v1-verification-r2.json'),
        ("verification=ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-verification'", "verification=ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-verification-r2'"),
        ("event_model['weights'])", "event_model['weights'],per_frame_seconds=10,max_seconds=600)"),
        ("    def persist():", "    result['local_verification_revision']=" + repr(provenance) + "\n    def persist():"),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=__file__))


if __name__ == '__main__':
    main()
