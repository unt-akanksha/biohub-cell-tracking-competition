"""Fixed held-embryo gate ablation: verified FP-only supervision, no feature changes."""
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_correction_gate_v1 import apply_mask, FEATURES
from research.trajectory_correction_logistic_v1 import fit, predict
from research.trajectory_correction_fp_labels_v2 import relabel


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    name = 'trajectory-correction-fp-gate-v2'
    target = ROOT / '.biohub/cache' / name
    receipt = ROOT / 'reports/experiments' / (name + '.json')
    assert not target.exists() and not receipt.exists(), 'Inspect existing attempt'
    folder = ROOT / '.biohub/cache/trajectory-correction-source-v1'
    dataset = read(folder / 'RESULT.json')
    assert sha(folder / 'RESULT.json') == 'e279e24348d17e60e07b8e6bb964ffab95490e502a3f70ac67ecd35928dfa1db'
    assert dataset['source_only'] and len(dataset['records']) == 60
    assert dataset['feature_names'] == list(FEATURES) and not dataset['features_include_ids']
    assert sha(ROOT / 'research/trajectory_correction_gate_v1.py') == dataset['gate_helper_sha256']
    audit_path = ROOT / 'reports/experiments/trajectory-correction-metric-labels-v2-known.json'
    audit = read(audit_path)
    assert audit['status'] == 'source_counterfactual_labels_audited' and audit['audited_components'] == 36
    assert audit['source_only'] and audit['all_fully_known_neutral_cases']
    assert not audit['selection_or_validation_opened'] and not audit['oracle_graphs_exported']
    assert audit['dataset_sha256'] == sha(folder / 'RESULT.json')
    old_path = ROOT / 'reports/experiments/trajectory-correction-gate-oof-v1.json'
    assert sha(old_path) == 'bf735591bfb02bc707d7ec85fc8dd3aaa5baf02563dead1052b51c92eea90616'
    old = read(old_path)
    assert sha(ROOT / 'research/trajectory_correction_logistic_v1.py') == old['model_helper_sha256']
    audit_choices = {r['stem']: r['indices'] for r in audit['plan']}
    assert set(audit_choices) == set(audit['records'])
    data, changed = {}, {}
    for stem, item in dataset['records'].items():
        fp, lp = folder / (stem + '-features.npz'), folder / (stem + '-labels.json')
        assert sha(fp) == item['features_sha256'] and sha(lp) == item['labels_sha256']
        with np.load(fp, allow_pickle=False) as values:
            assert list(values['feature_names']) == list(FEATURES)
            x = values['features'].copy()
        original = np.asarray([r['label'] for r in read(lp)], dtype=np.int8)
        components = audit['records'].get(stem, {}).get('components', [])
        assert [r['index'] for r in components] == audit_choices.get(stem, [])
        y = relabel(original, components)
        assert x.shape == (len(y), len(FEATURES))
        data[stem] = dict(x=x, old=original, y=y, embryo=item['embryo'])
        changed[stem] = dict(audited=len(components), positive=int(((y == 1) & (original == -1)).sum()),
                             negative=int(((y == 0) & (original == -1)).sum()))
    assert sum(r['audited'] for r in changed.values()) == 36
    target.mkdir()
    report = dict(status='fitting_source_folds', source_only=True, embryo_held_out_gate=True,
        source_sha256=sha(Path(__file__)), design_sha256=sha(ROOT / 'reports/experiments' / (name + '-design.md')),
        audit_sha256=sha(audit_path), old_gate_sha256=sha(old_path), dataset_sha256=sha(folder / 'RESULT.json'),
        label_helper_sha256=sha(ROOT / 'research/trajectory_correction_fp_labels_v2.py'),
        model_helper_sha256=old['model_helper_sha256'], regularization=.1, threshold=.5,
        threshold_search=False, feature_names=list(FEATURES), revised_labels=changed,
        prior_backbone_and_anchor_independence_not_established=True,
        selection_or_validation_opened=False, authorized_for_submission=False, models={}, inference={})

    def persist():
        report['seconds'] = time.monotonic() - start
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        (target / 'RESULT.json').write_text(text, encoding='utf-8')
        receipt.write_text(text, encoding='utf-8')

    persist()
    try:
        fitted, held_order = {}, {}
        for held in ('44b6', '6bba'):
            stems = sorted(s for s, r in data.items() if r['embryo'] != held)
            def training(key):
                return (np.concatenate([data[s]['x'][data[s][key] >= 0] for s in stems]),
                        np.concatenate([data[s][key][data[s][key] >= 0] for s in stems]))
            control = fit(*training('old'), regularization=.1)
            previous_path = ROOT / '.biohub/cache/trajectory-correction-gate-oof-v1' / ('held-out-' + held + '-model.json')
            assert sha(previous_path) == old['models'][held]['sha256']
            previous = read(previous_path)
            assert previous['training_stems'] == stems
            for key in ('mean', 'scale', 'coefficients', 'intercept', 'objective'):
                np.testing.assert_allclose(control[key], previous[key], rtol=0, atol=1e-10)
            model = fit(*training('y'), regularization=.1)
            model.update(held_out_embryo=held, training_stems=stems, feature_names=list(FEATURES))
            model_path = target / ('held-out-' + held + '-model.json')
            model_path.write_text(json.dumps(model, indent=2) + '\n', encoding='utf-8')
            report['models'][held] = dict(sha256=sha(model_path), old_fit_replayed_atol=1e-10,
                training_movies=len(stems), training_examples=model['training_examples'], class_counts=model['class_counts'])
            persist()
            fitted[held] = model
            held_order[held] = sorted((s for s, r in data.items() if r['embryo'] == held),
                key=lambda s: ((ROOT / dataset['records'][s]['baseline_path']).stat().st_size, s))
            print(json.dumps(dict(event='fold_fitted', held=held, **report['models'][held])), flush=True)
        order = [(held, stems[0]) for held, stems in held_order.items()]
        order += [(held, stem) for held, stems in held_order.items() for stem in stems[1:]]
        for index, (held, stem) in enumerate(order):
            model = fitted[held]
            assert stem not in model['training_stems']
            item = dataset['records'][stem]
            bp, cp = ROOT / item['baseline_path'], ROOT / item['candidate_path']
            assert sha(bp) == item['baseline_sha256'] and sha(cp) == item['candidate_sha256']
            base, candidate = read(bp), read(cp)
            initial_path = ROOT / item['initial_path']
            backup = read(ROOT / 'reports/experiments' / ('trajectory-event-source-v1-b' + str(item['batch']) + '-full-harvest.json'))
            assert backup['status'] == 'verified_backup'
            expected = next(r['sha256'] for r in backup['records'] if r['path'] == stem + '-original/pre-postprocess.json')
            assert sha(initial_path) == expected
            mask = predict(data[stem]['x'], model) >= .5
            graph = apply_mask(read(initial_path), base, candidate, mask)
            validate_graph(graph, 100)
            assert graph['nodes'] == base['nodes']
            output = target / (stem + '-prediction.json')
            output.write_text(json.dumps(graph, sort_keys=True, allow_nan=False), encoding='utf-8')
            report['inference'][stem] = dict(prediction_sha256=sha(output), accepted=int(mask.sum()),
                rejected=int((~mask).sum()), held_out_embryo=held, no_movie_labels_in_fit=True,
                gate_model_sha256=report['models'][held]['sha256'])
            if index == 1:
                report['two_complete_movie_smokes_passed'] = True
                print(json.dumps(dict(event='two_complete_movie_smokes_passed')), flush=True)
            persist()
        assert len(report['inference']) == 60
        report.update(status='all_predictions_frozen_before_scoring', all_predictions_frozen_before_scoring=True)
        persist()
        helper = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
        scorer = helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
        manifest = truth_root / 'train_geff_cache_manifest.json'
        assert sha(manifest) == '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files = read(manifest)['files']
        rows = dict(baseline=old['rows']['baseline'], anchor=old['rows']['anchor'], old_gate=old['rows']['gated'], revised_gate=[])
        for stem, item in dataset['records'].items():
            matches = [r for r in files if r['relative_path'].startswith(stem + '.geff/')]
            assert len(matches) == 21
            for r in matches:
                assert sha(truth_root / 'train' / r['relative_path']) == r['sha256']
            truth_path = truth_root / 'train' / (stem + '.geff')
            output = target / (stem + '-prediction.json')
            assert sha(output) == report['inference'][stem]['prediction_sha256']
            truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            graph = helper['prediction_graph'](read(output))
            evaluation = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
            count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            rows['revised_gate'].append(dict(scorer.per_sample_metrics(evaluation, count, scorer.node_recall(graph, truth)),
                                            stem=stem, embryo=item['embryo']))
            report.update(status='scoring_frozen_predictions', scored_rows=helper['finite'](rows['revised_gate']))
            persist()
            print(json.dumps(dict(event='revised_gate_scored', stem=stem)), flush=True)
        assert all(len(v) == 60 and len({r['stem'] for r in v}) == 60 for v in rows.values())
        summaries = {arm: scorer.summarise(values) for arm, values in rows.items()}
        embryos = {e: {a: scorer.summarise([r for r in v if r['embryo'] == e]) for a, v in rows.items()} for e in ('44b6', '6bba')}
        movies = {a: {r['stem']: scorer.summarise([r]) for r in v} for a, v in rows.items()}
        new, anchor = summaries['revised_gate'], summaries['anchor']
        regressions = sorted((dict(stem=s, delta=movies['revised_gate'][s]['score'] - movies['anchor'][s]['score'])
                              for s in data), key=lambda r: r['delta'])
        checks = dict(finite_complete=all(r['n'] == r['n_adj'] == 60 and np.isfinite(r['score']) for r in summaries.values()),
            pooled_beats_anchor=new['score'] > anchor['score'], pooled_beats_old_gate=new['score'] > summaries['old_gate']['score'],
            both_embryos_nonregressing=all(v['revised_gate']['score'] >= v['anchor']['score'] for v in embryos.values()),
            every_movie_nonregressing=all(r['delta'] >= -1e-12 for r in regressions),
            raw_edges_nonregressing=new['edge_jaccard'] >= anchor['edge_jaccard'],
            divisions_nonregressing=new['division_tp'] >= anchor['division_tp'] and new['division_fp'] <= anchor['division_fp'] and new['division_fn'] <= anchor['division_fn'])
        report.update(status='source_fp_gate_scored', rows=helper['finite'](rows), summaries=helper['finite'](summaries),
            by_embryo=helper['finite'](embryos), per_movie_summaries=helper['finite'](movies),
            per_movie_deltas=regressions, quality_checks=checks, eligible_for_separate_selection_test=all(checks.values()))
        persist()
        print(json.dumps(dict(status=report['status'], summaries=report['summaries'], checks=checks)), flush=True)
    except BaseException as error:
        report.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
