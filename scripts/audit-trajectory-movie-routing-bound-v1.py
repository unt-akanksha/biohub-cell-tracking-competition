"""Source-only perfect-choice upper bound, never an inference routing policy."""
import json
from pathlib import Path
import runpy
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_movie_routing_bound_v1 import fractional_bound


def main():
    prior_path = ROOT / 'reports/experiments/trajectory-correction-gate-oof-v1.json'
    assert sha(prior_path) == 'bf735591bfb02bc707d7ec85fc8dd3aaa5baf02563dead1052b51c92eea90616'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    assert prior['source_only'] and not prior['selection_or_validation_opened']
    output = ROOT / 'reports/experiments/trajectory-movie-routing-bound-v1.json'
    assert not output.exists()
    scorer = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))['load_scorer']()
    arms = {a: {r['stem']: r for r in prior['rows'][a]} for a in ('baseline', 'anchor')}
    assert set(arms['baseline']) == set(arms['anchor']) and len(arms['anchor']) == 60
    numerators, denominators = [], []
    for stem in sorted(arms['baseline']):
        base, anchor = (arms[a][stem] for a in ('baseline', 'anchor'))
        for key in ('num_pred_nodes', 'division_tp', 'division_fp', 'division_fn', 'total_node_ratio'):
            assert base[key] == anchor[key], (stem, key)
        assert base['edge_tp'] + base['edge_fn'] == anchor['edge_tp'] + anchor['edge_fn']
        factor = max(0., 1. - scorer.ADJUSTMENT_ALPHA * base['total_node_ratio'])
        numerators.append([r['edge_tp'] * factor for r in (base, anchor)])
        denominators.append([r['edge_tp'] + r['edge_fp'] + r['edge_fn'] for r in (base, anchor)])
    n, d = np.asarray(numerators), np.asarray(denominators)
    summaries = {a: scorer.summarise(list(values.values())) for a, values in arms.items()}
    division = summaries['anchor']['division_jaccard']
    fixed_division_term = scorer.SCORE_DIVISION_WEIGHT * division if np.isfinite(division) else 0.
    for i, arm in enumerate(arms):
        assert abs(n[:, i].sum() / d[:, i].sum() + fixed_division_term - summaries[arm]['score']) < 1e-12
    bound = fractional_bound(n, d)
    score = bound['ratio'] + fixed_division_term
    result = dict(status='source_whole_movie_routing_bound_computed', source_only=True,
        source_sha256=sha(Path(__file__)), helper_sha256=sha(ROOT / 'research/trajectory_movie_routing_bound_v1.py'),
        prior_results_sha256=sha(prior_path), movies=60, model_fitted=False, authorized_for_submission=False,
        selection_or_validation_opened=False, new_ground_truth_opened=False,
        uses_existing_ground_truth_derived_scores=True, diagnostic_oracle_only=True,
        routing_decisions_or_graphs_exported=False, applies_only_to_whole_movie_baseline_vs_fixed8_choice=True,
        not_an_upper_bound_on_local_routing_or_new_models=True, not_a_leaderboard_prediction=True,
        summaries=summaries, fractional_certificate=bound, fixed_division_term=fixed_division_term,
        perfect_choice_source_score_upper_bound=score, maximum_gain_over_anchor=score - summaries['anchor']['score'])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
