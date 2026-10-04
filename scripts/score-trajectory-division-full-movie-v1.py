"""Score all frozen division-positive graphs, then the joint eight-movie check."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha, csv_equivalent_graph
from research.trajectory_division_full_movie_v1 import STEMS
from research.trajectory_division_quality_v1 import check


def main(args):
    target = ROOT / 'reports/experiments/trajectory-division-full-movie-v1-result.json'
    if target.exists():
        raise ValueError('Preserve completed scores')
    folder = ROOT / '.biohub/cache/trajectory-division-full-v1-output'
    terminal_path = folder / 'result.json'
    if sha(terminal_path) != args.terminal_sha256:
        raise ValueError('Terminal hash mismatch')
    terminal = json.loads(terminal_path.read_text())
    contract_path = ROOT / '.biohub/cache/trajectory-division-full-movie-v1-bundle/CONTRACT.json'
    contract_sha = '7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1'
    if (sha(contract_path) != contract_sha or terminal['contract_sha256'] != contract_sha
            or terminal['status'] != 'complete_prelabel_predictions' or terminal['mode'] != 'full'
            or not terminal['inputs_unchanged'] or terminal['ground_truth_opened'] is not False
            or terminal['authorized_for_submission'] is not False
            or not 0 < terminal['elapsed_seconds'] <= 3600 or sorted(terminal['movies']) != list(STEMS)):
        raise ValueError('Complete same-contract label-free full run required')
    transport = json.loads((folder / 'REMOTE_ARTIFACT_MANIFEST.json').read_text())
    for row in transport['files']:
        if sha(folder / row['path']) != row['sha256']:
            raise ValueError('Downloaded inference artifact changed')
    prepared = {}
    for stem in STEMS:
        record = terminal['movies'][stem]['original']
        if record['frames'] != 100 or set(terminal['movies'][stem]) != {'original'}:
            raise ValueError('One full public inference required per movie')
        prepared[stem] = {}
        for arm, filename, field in (('original','prediction.json','prediction_sha256'),
                                      ('repaired','repaired-prediction.json','repaired_sha256')):
            path = folder / (stem + '-original') / filename
            if sha(path) != record[field]:
                raise ValueError('Prediction changed after label-free freeze')
            graph = json.loads(path.read_text())
            if csv_equivalent_graph({int(k):v for k,v in graph['nodes'].items()}, graph['edges'], 100) != graph:
                raise ValueError('Invalid full movie graph')
            prepared[stem][arm] = graph
        base, repaired = prepared[stem]['original'], prepared[stem]['repaired']
        old_edges = {(e['source_id'],e['target_id']) for e in base['edges']}
        new_edges = {(e['source_id'],e['target_id']) for e in repaired['edges']}
        if base['nodes'] != repaired['nodes'] or not old_edges <= new_edges or len(new_edges-old_edges) != record['added_edges']:
            raise ValueError('Candidate altered old output or wrong edge count')
    # Validate exact score-reuse provenance before any new annotations are opened.
    baseline_path = ROOT / 'reports/experiments/public-d4-full-movie-v1-result.json'
    cv_path = ROOT / 'reports/experiments/learned-trajectory-endpoint-v1-result.json'
    mixture_path = ROOT / 'reports/experiments/trajectory-source-mixture-v1-diagnostic.json'
    for path, expected in ((baseline_path,'04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b'),
                           (cv_path,'f48299001d0733c601b8356c2977dffd2ac79c9cd829252d7b504d0741e0a8d1'),
                           (mixture_path,'4216fa5bb693957806eaa04e25f3c0b297e5507674bac754f649edca6e1dd6f3')):
        if sha(path) != expected:
            raise ValueError('Earlier exact-graph evidence changed')
    mixture = json.loads(mixture_path.read_text())
    if mixture['status'] != 'identical_diagnostic_pass':
        raise ValueError('Prior mixture diagnostic did not pass')
    truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
    manifest_path = truth_root / 'train_geff_cache_manifest.json'
    if sha(manifest_path) != '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':
        raise ValueError('Original truth inventory changed')
    manifest = json.loads(manifest_path.read_text()); truth_receipts = []
    for stem in STEMS:
        records = [r for r in manifest['files'] if r['relative_path'].startswith(stem + '.geff/')]
        observed = {p.relative_to(truth_root / 'train').as_posix()
                    for p in (truth_root / 'train' / (stem+'.geff')).rglob('*') if p.is_file()}
        if len(records) != 21 or observed != {r['relative_path'] for r in records}:
            raise ValueError('Full original GEFF inventory required')
        for row in records:
            path = truth_root / 'train' / row['relative_path']
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Truth changed')
        truth_receipts.extend(records)
    v1 = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    scorer = v1['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    additional = {a:[] for a in ('original','repaired')}
    for stem in STEMS:
        for arm in additional:
            graph = v1['prediction_graph'](prepared[stem][arm])
            path = truth_root / 'train' / (stem + '.geff')
            truth = td.graph.IndexedRXGraph.from_geff(str(path))[0]
            er = scorer.evaluate(graph, truth, scale=(1.625,.40625,.40625), max_distance=7.)
            count = float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
            row = dict(scorer.per_sample_metrics(er, count, scorer.node_recall(graph, truth)),
                       stem=stem, embryo=stem.split('_')[0])
            additional[arm].append(row)
            print(json.dumps(v1['finite'](dict(event='additional_movie_scored', arm=arm, **row))), flush=True)
    baseline = json.loads(baseline_path.read_text()); cv = json.loads(cv_path.read_text())
    rows = dict(original=baseline['per_movie']['original'] + additional['original'],
                repaired=cv['per_movie'] + additional['repaired'])
    summaries = {a:scorer.summarise(v) for a,v in rows.items()}
    embryos = {a:{e:scorer.summarise([r for r in v if r['embryo']==e]) for e in ('44b6','6bba')} for a,v in rows.items()}
    movies = {a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
    comparison = check(rows, summaries, embryos, movies, STEMS)
    result = dict(status='diagnostic_pass' if comparison['diagnostic_gate_passed'] else 'rejected_diagnostic',
                  run_id='trajectory-division-full-movie-v1', source_sha256=sha(Path(__file__)),
                  quality_source_sha256=sha(ROOT / 'research/trajectory_division_quality_v1.py'),
                  contract_sha256=contract_sha, terminal_sha256=args.terminal_sha256,
                  verified_truth_files=truth_receipts, additional_movies=additional,
                  joint_eight_movie_rows=rows, summaries=summaries, by_embryo=embryos,
                  per_movie_summaries=movies, comparison=comparison,
                  independently_held_out=False, public_backbone_training_overlap=True,
                  authorized_for_submission=False, authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    target.write_text(json.dumps(v1['finite'](result), indent=2, allow_nan=False) + '\n')
    print(json.dumps(v1['finite'](dict(status=result['status'], summaries=summaries, comparison=comparison))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--terminal-sha256', required=True)
    main(p.parse_args())
