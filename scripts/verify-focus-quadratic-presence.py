"""Replay actual saved LOMO models and their fitting-only normalization."""
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
from research.focus_parent_presence import metrics
from research.focus_quadratic_presence import evaluate, expand, safe_std, screen


def verify():
    runner = runpy.run_path(str(ROOT / 'scripts/fit-focus-quadratic-presence.py'))
    sha, run = runner['sha'], runner['RUN']
    path = ROOT / f'reports/experiments/{run}-result.json'
    result = json.loads(path.read_text())
    for relative, digest in result['source_hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Actual executed method changed')
    movies, evidence = runner['load_fitting']()
    if result['evidence'] != evidence or [f['held_out'] for f in result['folds']] != list(movies):
        raise ValueError('Exact twelve-movie fitting provenance required')
    cache = ROOT / '.biohub/cache' / run
    for fold in result['folds']:
        held = fold['held_out']
        stems = [stem for stem in movies if stem != held]
        data = runner['concatenate'](movies, stems)
        model_path = cache / (held + '-models.json')
        saved = json.loads(model_path.read_text())
        if (sha(model_path) != fold['models_sha256'] or saved['held_out'] != held
                or saved['fitting_stems'] != stems or fold['fitting_stems'] != stems
                or json.loads((cache / (held + '-result.json')).read_text()) != fold):
            raise ValueError('Actual saved fold identity differs')
        x = data['context']
        for arm in ('linear', 'quadratic'):
            model = saved[arm]
            if (model['fitting_examples'] != len(x) or model['fitting_present'] != int(data['present'].sum())
                    or not np.array_equal(model['mean'], x.mean(axis=0))):
                raise ValueError('Training-only fitted scope/normalization differs')
        quadratic = saved['quadratic']
        scale = safe_std(x)
        basis = expand((x - x.mean(axis=0)) / scale)
        if (not np.array_equal(quadratic['scale'], scale)
                or not np.array_equal(saved['linear']['std'], scale)
                or not np.array_equal(quadratic['basis_mean'], basis.mean(axis=0))
                or not np.array_equal(quadratic['basis_scale'], safe_std(basis))):
            raise ValueError('Held-out information entered normalization')
        replay = dict(original=metrics(movies[held]), linear=metrics(movies[held], saved['linear']),
                      quadratic=evaluate(movies[held], quadratic))
        for arm, row in replay.items():
            for key, value in row.items():
                if abs(value - fold[arm][key]) > 1e-10:
                    raise ValueError('Actual saved posterior/decision replay differs')
    gate = screen(result['folds'])
    if gate != result['screening_gate'] or result['gpu_seconds'] != 0 or result['new_target_movies_opened'] != 0:
        raise ValueError('Exact screen and resource scope required')
    if any(result[k] is not False for k in ('diagnostic_evaluated', 'source_selection_opened', 'final_model_fitted', 'authorized_for_submission')):
        raise ValueError('Fitting-only screen scope required')
    all_data = runner['concatenate'](movies, list(movies))
    conditional = float((all_data['conditional_nll'] * all_data['present']).sum())
    original_loss = gate['pooled']['original']['loss_sum']
    ranking = dict(conditional_parent_nll_sum=conditional,
                   conditional_fraction_of_original_nll=conditional / original_loss,
                   original_joint_nll_sum=original_loss,
                   oracle_presence_nll_lower_bound=conditional / len(all_data['present']),
                   parent_ranking_correct_ceiling=int(all_data['correct_if_present'].sum()),
                   known_parent=int(all_data['present'].sum()),
                   caveat='Analytical bound for the fixed original parent ranking only; fitting summaries, not tracking score')
    return dict(status='verified_twelve_movie_quadratic_lomo', result_sha256=sha(path),
                all_twelve_saved_models_replayed=True, fitting_only_normalization_replayed=True,
                screening_gate=gate, ranking_error_budget=ranking,
                diagnostic_arrays_opened=False, authorized_for_submission=False)


if __name__ == '__main__':
    path = ROOT / 'reports/experiments/focus-quadratic-presence-v1-verification.json'
    if path.exists():
        raise ValueError('Never overwrite verified receipt')
    result = verify()
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
