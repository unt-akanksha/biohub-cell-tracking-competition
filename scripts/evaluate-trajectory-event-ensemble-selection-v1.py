"""Frozen three-arm event-head selection; no auto promotion or GPU launch."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_event_release_v1 import check_event_stage
from research.trajectory_event_ensemble_selection_v1 import ARMS, fixed_weights, quality_checks, select_candidate

NAME = 'trajectory-event-ensemble-selection-v1'
REPORTS = ROOT / 'reports/experiments'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def inputs_and_contract():
    evaluate = runpy.run_path(str(ROOT / 'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    fits = evaluate['fixed_fit_contracts'](ROOT)
    training_stems = set()
    for embryo in fits:
        contract = read(ROOT / '.biohub/cache' / ('trajectory-event-fork16-fit-' + embryo + '-v1') / 'CONTRACT.json')
        training_stems.update(contract['source_stems'])
    assert len(training_stems) == 60
    folder = ROOT / '.biohub/cache/trajectory-event-selection-v1-features'
    assert sha(folder / 'RESULT.json') == '75258e792ca5eeb2d9763c20a43d2478ee55aa31ddbef0ef2ea596aa4f660d11'
    frozen = read(folder / 'RESULT.json')
    scope = ROOT / '.biohub/cache/trajectory-ranker-selection-v1-plan/MOVIES.json'
    assert sha(scope) == frozen['scope_sha256']
    assert frozen['source_role_disjoint'] and not frozen['selection_labels_used']
    movies = read(scope)['movies']
    assert len(movies) == 10 and all(m['role'] == 'selection' for m in movies)
    assert not training_stems.intersection(m['stem'] for m in movies)
    old_path = REPORTS / 'trajectory-event-anchor-selection-v1.json'
    assert sha(old_path) == '43e94177fe98c2ade5925aa78fd5c892a420a7d8323d1dfa986c75885d240fcb'
    old = read(old_path)
    rows = {}
    for movie in movies:
        stem = movie['stem']
        paths = {k: folder / (stem + '-' + k + ('.json' if k == 'prediction' else '.npz'))
                 for k in ('prediction', 'features', 'groups')}
        for key, path in paths.items():
            assert sha(path) == frozen['per_movie'][stem][key + '_sha256']
        paths['initial'] = ROOT / '.biohub/cache/trajectory-ranker-selection-v1-full-output' / (stem + '-original/pre-postprocess.json')
        assert sha(paths['initial']) == frozen['per_movie'][stem]['initial_sha256']
        paths['submitted8'] = ROOT / '.biohub/cache/trajectory-event-anchor-selection-v1' / (stem + '-prediction.json')
        assert sha(paths['submitted8']) == old['inference'][stem]['prediction_sha256']
        rows[stem] = dict(paths=paths, hashes={k: sha(p) for k, p in paths.items()},
                          feature_bytes=paths['features'].stat().st_size)
    proof_path = REPORTS / 'trajectory-event-null-portable-v2-full.json'
    assert sha(proof_path) == '81d90e28b45d344e1b30afe195958cf02fcf967d15b3c3a801b2b6cd2694a13f'
    proof = read(proof_path)
    assert proof['status'] == 'standalone_exact_replay_passed' and len(proof['records']) == 4
    runtime_path = ROOT / '.biohub/cache/trajectory-event-null-portable-v2-full/event-trajectory.py'
    assert sha(runtime_path) == proof['code_sha256']
    smoke_movies = [min((s for s in rows if s.startswith(e + '_')), key=lambda s: (rows[s]['feature_bytes'], s))
                    for e in ('44b6', '6bba')]
    contract = dict(fit_contracts=fits, movies=sorted(rows), smoke_movies=smoke_movies,
        input_hashes={s: v['hashes'] for s, v in rows.items()}, source_sha256=sha(Path(__file__)),
        design_sha256=sha(REPORTS / (NAME + '-design.md')),
        helper_sha256=sha(ROOT / 'research/trajectory_event_ensemble_selection_v1.py'),
        graph_checker_sha256=sha(ROOT / 'research/trajectory_event_release_v1.py'),
        runtime_sha256=sha(runtime_path), arms=list(ARMS), mean_coefficients=[.5, .5],
        per_frame_seconds=10., per_movie_seconds=600., source_role_disjoint=True,
        prior_pipeline_selection_exposure_disclosed=True, source44_failed_transfer_disclosed=True,
        movie_id_routing=False, training_schedule_changed=False, authorized_for_submission=False)
    return rows, contract, runtime_path, old


def main(mode):
    started = time.monotonic()
    inputs, contract, runtime_path, old = inputs_and_contract()
    contract_path = REPORTS / (NAME + '-contract.json')
    if mode == 'freeze':
        assert not contract_path.exists(), 'Do not replace the frozen protocol'
        current = read(REPORTS / 'trajectory-event-fork16-fit-6bba-v1.json')
        assert current['status'] == 'running' and not current['model_fitted']
        write(contract_path, dict(contract=contract, source6_fit_status_at_freeze=current['status'],
            source6_steps_at_freeze=current['steps'], source6_report_sha256_at_freeze=sha(REPORTS / 'trajectory-event-fork16-fit-6bba-v1.json')))
        print(json.dumps(dict(status='protocol_frozen_before_second_final_weights', sha256=sha(contract_path))))
        return
    assert read(contract_path)['contract'] == contract, 'Frozen implementation or input changed'
    members, model_proofs = {}, {}
    for embryo in ('44b6', '6bba'):
        root = ROOT / '.biohub/cache' / ('trajectory-event-fork16-fit-' + embryo + '-v1')
        result_path = REPORTS / ('trajectory-event-fork16-fit-' + embryo + '-v1.json')
        result, fit_contract = read(result_path), read(root / 'CONTRACT.json')
        assert result['status'] == 'source_event_training_complete' and result['model_fitted']
        assert result['steps'] == fit_contract['cases'] * fit_contract['epochs']
        assert fit_contract['epochs'] == len(result['epochs_completed']) == 3
        weights_path = root / 'final-weights.npz'
        assert sha(weights_path) == result['weights_sha256']
        data = arrays(weights_path)
        assert int(data['max_fork_children']) == 16
        members[embryo] = data
        model_proofs[embryo] = dict(weights_sha256=sha(weights_path), fit_report_sha256=sha(result_path))
    assert list(members['44b6']['feature_names']) == list(members['6bba']['feature_names'])
    weights = fixed_weights(members['44b6']['weights'], members['6bba']['weights'])
    name = NAME + ('-smoke' if mode == 'smoke' else '')
    receipt, target = REPORTS / (name + '.json'), ROOT / '.biohub/cache' / name
    assert not receipt.exists() and not target.exists(), 'Inspect prior run; do not restart blindly'
    smoke = None
    if mode == 'full':
        smoke = read(REPORTS / (NAME + '-smoke.json'))
        assert smoke['status'] == 'selection_inference_smoke_passed'
        assert smoke['contract_sha256'] == sha(contract_path) and smoke['model_proofs'] == model_proofs
        assert not smoke['ground_truth_opened']
    runtime = runpy.run_path(str(runtime_path))
    assert list(members['44b6']['feature_names']) == list(runtime['FEATURES'])
    movies = contract['smoke_movies'] if mode == 'smoke' else contract['movies']
    target.mkdir()
    write(target / 'MODELS.json', {a: w.tolist() for a, w in weights.items()})
    report = dict(status='freezing_predictions', contract_sha256=sha(contract_path),
        model_proofs=model_proofs, models_sha256=sha(target / 'MODELS.json'), movies=movies,
        inference={}, ground_truth_opened=False, all_predictions_frozen_before_scoring=False,
        gpu_used=False, training_changed=False, authorized_for_submission=False,
        independent_validation_authorized=False, prior_pipeline_selection_exposure_disclosed=True)

    def persist():
        report['seconds'] = time.monotonic() - started
        write(receipt, report)
        write(target / 'RESULT.json', report)

    persist()
    try:
        for stem in movies:
            paths = inputs[stem]['paths']
            initial, baseline = read(paths['initial']), read(paths['prediction'])
            groups, features = arrays(paths['groups']), arrays(paths['features'])
            assert list(features['feature_names']) == list(members['44b6']['feature_names'][:18])
            report['inference'][stem] = {}
            for arm in ARMS:
                reused = smoke is not None and stem in smoke['inference']
                if reused:
                    record = smoke['inference'][stem][arm]
                    prior = ROOT / '.biohub/cache' / (NAME + '-smoke') / (stem + '-' + arm + '.json')
                    assert sha(prior) == record['prediction_sha256']
                    graph, details = read(prior), record['details']
                else:
                    graph, details = runtime['refine_prepared'](initial, baseline, groups, features['features'],
                        weights[arm], per_frame_seconds=10., max_seconds=600.)
                validate_graph(graph, 100)
                check_event_stage(initial, baseline, graph, details, frames=100)
                assert details['processed_frames'] == 99 and not details['budget_exhausted'] and not details['solver_fallbacks']
                output = target / (stem + '-' + arm + '.json')
                write(output, graph)
                report['inference'][stem][arm] = dict(prediction_sha256=sha(output), details=details, reused_smoke=reused)
                persist()
            print(json.dumps(dict(event='selection_movie_frozen', stem=stem)), flush=True)
        report['all_predictions_frozen_before_scoring'] = True
        if mode == 'smoke':
            report['status'] = 'selection_inference_smoke_passed'
            persist()
            return
        helper = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
        scorer = helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
        manifest = truth_root / 'train_geff_cache_manifest.json'
        assert sha(manifest) == '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files = read(manifest)['files']
        rows = {a: [] for a in ('submitted8',) + ARMS}
        report.update(status='scoring_frozen_predictions', ground_truth_opened=True)
        persist()
        for stem in movies:
            gt_files = [r for r in files if r['relative_path'].startswith(stem + '.geff/')]
            assert len(gt_files) == 21
            for r in gt_files:
                assert sha(truth_root / 'train' / r['relative_path']) == r['sha256']
            gt_path = truth_root / 'train' / (stem + '.geff')
            for arm in rows:
                path = inputs[stem]['paths']['submitted8'] if arm == 'submitted8' else target / (stem + '-' + arm + '.json')
                expected = inputs[stem]['hashes']['submitted8'] if arm == 'submitted8' else report['inference'][stem][arm]['prediction_sha256']
                assert sha(path) == expected
                truth = td.graph.IndexedRXGraph.from_geff(str(gt_path))[0]
                graph = helper['prediction_graph'](read(path))
                scored = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
                count = float(GeffMetadata.read(str(gt_path)).extra['estimated_number_of_nodes'])
                row = dict(scorer.per_sample_metrics(scored, count, scorer.node_recall(graph, truth)), stem=stem, embryo=stem.split('_')[0])
                if arm == 'submitted8':
                    assert abs(scorer.summarise([row])['score'] - old['per_movie_summaries']['anchor'][stem]['score']) < 1e-12
                rows[arm].append(row)
            print(json.dumps(dict(event='selection_movie_scored', stem=stem)), flush=True)
        summaries = {a: scorer.summarise(v) for a, v in rows.items()}
        per_movie = {a: {r['stem']: scorer.summarise([r]) for r in v} for a, v in rows.items()}
        by_embryo = {e: {a: scorer.summarise([r for r in v if r['embryo'] == e]) for a, v in rows.items()} for e in ('44b6', '6bba')}
        checks = quality_checks(summaries, per_movie, by_embryo, movies)
        assert inputs_and_contract()[1] == contract
        report.update(status='fixed_ensemble_selection_scoring_complete', rows=helper['finite'](rows),
            summaries=helper['finite'](summaries), per_movie_summaries=helper['finite'](per_movie),
            by_embryo=helper['finite'](by_embryo), quality_checks=checks,
            selected_candidate=select_candidate(summaries, checks),
            regressions={a: {s: per_movie[a][s]['score'] - per_movie['submitted8'][s]['score'] for s in movies
                if per_movie[a][s]['score'] < per_movie['submitted8'][s]['score'] - 1e-12} for a in ARMS})
        persist()
        print(json.dumps(dict(status=report['status'], summaries=report['summaries'], checks=checks,
                              selected_candidate=report['selected_candidate'])), flush=True)
    except BaseException as error:
        report.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'smoke', 'full'))
    with threadpool_limits(limits=1, user_api='blas'):
        main(parser.parse_args().mode)
