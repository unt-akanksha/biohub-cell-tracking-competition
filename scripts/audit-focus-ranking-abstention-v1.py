"""Fitting-only error bound: can physical abstention preserve the ranking gain?

No optimization or new predictor export. Replays the old twelve stored LOMO
rankers, then counts whether their true-parent ranking gains survive retaining
every original physical null decision. This is a bound/error attribution, not
a source, diagnostic, threshold search, or submission acceptance experiment.
"""
import json
import os
from pathlib import Path
import runpy
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import metrics, validate


def main():
    target = ROOT / 'reports/experiments/focus-ranking-abstention-v1-audit.json'
    if target.exists():
        raise ValueError('Never overwrite completed error attribution')
    runner = runpy.run_path(str(ROOT / 'scripts/fit-focus-candidate-ranker.py'))
    sha = runner['sha']
    result_path = ROOT / 'reports/experiments/focus-candidate-ranker-v1-lomo.json'
    if sha(result_path) != '1682bf7167f5119bad272a26f035b15a1cfe25ae4c4ce6db7cc4a13a30446c92':
        raise ValueError('Actual original ranking experiment required')
    prior = json.loads(result_path.read_text())
    if any(sha(ROOT / p) != value for p, value in prior['source_hashes'].items()):
        raise ValueError('Frozen original ranking method changed')
    evidence, samples = runner['load']()
    if [r['held_out'] for r in prior['folds']] != list(samples):
        raise ValueError('All original fitting-only folds required')
    rows = []
    for record in prior['folds']:
        stem = record['held_out']
        data = samples[stem]
        path = ROOT / '.biohub/cache/focus-candidate-ranker-v1/lomo' / (stem + '-model.json')
        saved = json.loads(path.read_text())
        if (sha(path) != record['model_sha256'] or saved['held_out'] != stem
                or saved['training_stems'] != [s for s in samples if s != stem]):
            raise ValueError('Actual held-out model identity differs')
        if metrics(data, saved['model']) != record['candidate'] or metrics(data) != record['physical']:
            raise ValueError('Exact original predictions must replay before attribution')
        ids = validate(data)
        row_ids = np.arange(len(data['offset']))
        physical_max = np.maximum.reduceat(data['offset'], data['starts'])
        physical_choice = np.minimum.reduceat(
            np.where(data['offset'] == physical_max[ids], row_ids, len(row_ids)), data['starts'])
        physical_reject = physical_choice == data['null_rows']
        score = data['offset'] + data['features'] @ np.asarray(saved['model']['theta'])
        score[data['null_rows']] = -np.inf
        real_max = np.maximum.reduceat(score, data['starts'])
        real_choice = np.minimum.reduceat(
            np.where(np.isfinite(score) & (score == real_max[ids]), row_ids, len(row_ids)), data['starts'])
        present = data['present'].astype(bool)
        real_correct = present & (real_choice == data['chosen'])
        rows.append(dict(stem=stem, model_sha256=sha(path),
                         known_parent=int(present.sum()), known_absent=int((~present).sum()),
                         physical_parent_rejections=int((present & physical_reject).sum()),
                         physical_correct_absent=int((~present & physical_reject).sum()),
                         real_ranking_correct_parent=int(real_correct.sum()),
                         ranking_correct_but_physically_rejected=int((real_correct & physical_reject).sum()),
                         existing_ranking_with_physical_veto_correct_parent=int((real_correct & ~physical_reject).sum()),
                         oracle_ranking_with_physical_veto_parent_ceiling=int((present & ~physical_reject).sum())))
    keys = [k for k in rows[0] if k not in ('stem', 'model_sha256')]
    total = {k: sum(r[k] for r in rows) for k in keys}
    if (total['known_parent'] != 10754 or total['known_absent'] != 161
            or total['physical_correct_absent'] != 115
            or total['real_ranking_correct_parent'] - total['ranking_correct_but_physically_rejected']
            != total['existing_ranking_with_physical_veto_correct_parent']):
        raise ValueError('Complete count identities or physical-null replay differ')
    result = dict(status='completed_fitting_only_ranking_abstention_bound',
                  method_sha256=sha(Path(__file__)), prior_result_sha256=sha(result_path),
                  data_smoke_sha256=sha(ROOT / 'reports/experiments/focus-candidate-ranker-v1-data-smoke.json'),
                  complete_groups=evidence['total_groups'], complete_choices=evidence['total_choices'],
                  rows=rows, totals=total, parent_requirement=9835,
                  existing_ranking_physical_veto_reaches_parent_requirement=
                  total['existing_ranking_with_physical_veto_correct_parent'] >= 9835,
                  any_ranking_physical_veto_can_reach_parent_requirement=
                  total['oracle_ranking_with_physical_veto_parent_ceiling'] >= 9835,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  optimizer_run=False, gpu_seconds=0, authorized_for_submission=False)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
