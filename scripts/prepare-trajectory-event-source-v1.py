"""Freeze source-only full-movie prediction collection for structured events."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory_path = ROOT / 'reports/experiments/native-division-coverage-v3.json'
    roles_path = ROOT / '.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    cached_path = ROOT / '.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(inventory_path) == '5e674aad7607930cbb2f0e96d204d8bd13931a6fd3f407ddea1bc8d0e49f8cbe'
    assert sha(roles_path) == '60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    assert sha(cached_path) == '5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    roles = {m['stem']: m for m in json.loads(roles_path.read_text())['movies']}
    inventory = json.loads(inventory_path.read_text())['movies']
    event_movies = {m['stem']: m for m in inventory if m['role'] == 'optimization' and m['events']}
    cached = {m['stem'] for m in json.loads(cached_path.read_text())['movies']}
    selected = set(event_movies) | cached
    assert all(roles[s]['role'] == 'optimization' for s in selected)
    assert len(event_movies) == 63 and len(cached) == 8
    new = sorted(selected - cached, key=lambda s: hashlib.sha256(('trajectory-event-source-v1:' + s).encode()).hexdigest())
    assert not (set(new) & cached)
    target = ROOT / '.biohub/cache/trajectory-event-source-v1-plan'
    target.mkdir(exist_ok=False)
    batches = []
    for index, offset in enumerate(range(0, len(new), 8)):
        stems = new[offset:offset + 8]
        batch = target / ('batch-' + str(index))
        batch.mkdir()
        plan = dict(run_id='trajectory-event-source-v1-batch-' + str(index),
                    movies=[dict(stem=s, embryo=roles[s]['embryo'], role='optimization',
                                 image_frames=list(range(100))) for s in stems],
                    source_role_plan_sha256=sha(roles_path), ground_truth_included=False,
                    competition_test_data_read=False, selection_or_validation_opened=False,
                    purpose='Unchanged licensed predictor and AR2 graphs for future structured event fitting',
                    model_training_performed=False, authorized_for_submission=False)
        (batch / 'MOVIES.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        batches.append(dict(index=index, movies=stems, plan_sha256=sha(batch / 'MOVIES.json'),
                            maximum_image_cache_bytes=5_500_000_000, full_wall_cap_seconds=2700,
                            smoke_required=True, gpu_launched=False))
    counts = Counter()
    for s in selected:
        counts[roles[s]['embryo'] + '_movies'] += 1
        counts[roles[s]['embryo'] + '_events'] += len(event_movies.get(s, {}).get('events', []))
    result = dict(status='source_collection_scope_frozen_not_launched',
                  source_role_plan_sha256=sha(roles_path), division_inventory_sha256=sha(inventory_path),
                  reused_source_scope_sha256=sha(cached_path), selected_movies=sorted(selected),
                  reused_prediction_movies=sorted(cached), new_prediction_movies=new,
                  counts=dict(counts), batches=batches,
                  source_only=True, ground_truth_files_opened=False, gpu_launched=False,
                  selection_or_validation_opened=False, training_or_submission_performed=False,
                  active_kaggle_experiments_must_finish_before_gpu_collection=True,
                  frozen_submitted_evaluation_is_not_a_parallel_experiment=True,
                  foreign_jobs_must_not_be_interrupted=True,
                  each_batch_requires_verified_local_output_backup_before_ram_retirement=True,
                  first_batch_requires_functional_smoke_and_measured_full_runtime=True,
                  model_family='prospective event-structured graph learner, not rejected v3/v5/v6 ensemble',
                  predicted_graphs_or_model_scores_used_to_select_new_movies=False)
    (target / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], counts=result['counts'],
                         new_movies=len(new), batches=len(batches), reused_movies=len(cached))))


if __name__ == '__main__':
    main()
