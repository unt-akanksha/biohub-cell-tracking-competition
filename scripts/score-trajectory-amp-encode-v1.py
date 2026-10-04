"""Frozen four-movie AMP speed screen against the accepted FP32 candidate."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_division_full_movie_v1 import STEMS


def pinned(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen evidence changed: ' + path.name)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal-sha256', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    target = ROOT / 'reports/experiments/trajectory-amp-encode-v1-result.json'
    if target.exists():
        raise ValueError('Preserve completed precision screen')
    folder = ROOT / '.biohub/cache/trajectory-amp-encode-full-v1-output'
    terminal = pinned(folder / 'result.json', args.terminal_sha256)
    contract = '94fd57e631523de95201ec7ac57567ff1e2908a35b3bf9d6294fc2e1b6c46254'
    if (terminal['status'] != 'complete_prelabel_predictions'
            or terminal['contract_sha256'] != contract or not terminal['inputs_unchanged']
            or terminal['ground_truth_opened'] or terminal['mode'] != 'full'
            or set(terminal['movies']) != set(STEMS)
            or not 0 < terminal['elapsed_seconds'] <= 3600):
        raise ValueError('Complete same-contract full-movie predictions required')
    transport = json.loads((folder / 'REMOTE_ARTIFACT_MANIFEST.json').read_text())
    for record in transport['files']:
        path = folder / record['path']
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Transport receipt mismatch')
    prepared = {}
    for stem in STEMS:
        record = terminal['movies'][stem]['original']
        if record['frames'] != 100:
            raise ValueError('Full100-frame movie required')
        prepared[stem] = pinned(folder / (stem+'-original') / 'repaired-prediction.json',
                                record['repaired_sha256'])
        validate_graph(prepared[stem], 100)
    prior = pinned(ROOT / 'reports/experiments/trajectory-division-full-movie-v1-result.json',
                   '6dc733376a89e0eb8fdb8b996caf008252c84799199d35ef5f7f48a6ff39bb40')
    baseline = [r for r in prior['joint_eight_movie_rows']['repaired'] if r['stem'] in STEMS]
    if len(baseline) != 4:
        raise ValueError('All four accepted FP32 reference scores required')
    truth_root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1'
    manifest = pinned(truth_root / 'train_geff_cache_manifest.json',
                      '744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9')
    for stem in STEMS:
        records = [r for r in manifest['files'] if r['relative_path'].startswith(stem+'.geff/')]
        observed = {p.relative_to(truth_root/'train').as_posix()
                    for p in (truth_root/'train'/(stem+'.geff')).rglob('*') if p.is_file()}
        if len(records) != 21 or observed != {r['relative_path'] for r in records}:
            raise ValueError('Complete truth inventory required')
        for record in records:
            path = truth_root/'train'/record['relative_path']
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('Truth changed')
    # All predictions were frozen and checked before opening any annotations.
    helper = runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
    scorer = helper['load_scorer']()
    import tracksdata as td
    from geff import GeffMetadata
    rows = dict(fp32=baseline, amp=[])
    for stem in STEMS:
        truth_path = truth_root/'train'/(stem+'.geff')
        truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
        graph = helper['prediction_graph'](prepared[stem])
        evaluated = scorer.evaluate(graph, truth, scale=(1.625,.40625,.40625), max_distance=7.)
        count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
        row = dict(scorer.per_sample_metrics(evaluated, count, scorer.node_recall(graph,truth)),
                   stem=stem, embryo=stem.split('_')[0])
        rows['amp'].append(row)
        print(json.dumps(helper['finite'](dict(event='frozen_amp_scored', **row))), flush=True)
    summaries = {a:scorer.summarise(v) for a,v in rows.items()}
    movies = {a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
    embryos = {a:{e:scorer.summarise([r for r in v if r['embryo']==e])
                  for e in ('44b6','6bba')} for a,v in rows.items()}
    indexed = {a:{r['stem']:r for r in v} for a,v in rows.items()}
    movie_delta = {s:movies['amp'][s]['score']-movies['fp32'][s]['score'] for s in STEMS}
    embryo_delta = {e:embryos['amp'][e]['score']-embryos['fp32'][e]['score'] for e in ('44b6','6bba')}
    gates = dict(pooled_score_nonregress=summaries['amp']['score']>=summaries['fp32']['score'],
                 pooled_raw_edge_nonregress=summaries['amp']['edge_jaccard']>=summaries['fp32']['edge_jaccard'],
                 every_movie_nonregress=min(movie_delta.values())>=0,
                 both_embryos_nonregress=min(embryo_delta.values())>=0,
                 division_counts_nonregress=all(
                     indexed['amp'][s]['division_tp']>=indexed['fp32'][s]['division_tp']
                     and indexed['amp'][s]['division_fp']<=indexed['fp32'][s]['division_fp']
                     and indexed['amp'][s]['division_fn']<=indexed['fp32'][s]['division_fn'] for s in STEMS))
    result = dict(status='screen_passed_requires_eight_movie_validation' if all(gates.values()) else 'rejected_quality_regression',
                  contract_sha256=contract, terminal_sha256=args.terminal_sha256,
                  source_sha256=sha(Path(__file__)), rows=rows, summaries=summaries,
                  per_movie_summaries=movies, by_embryo=embryos, per_movie_delta=movie_delta,
                  per_embryo_delta=embryo_delta, gates=gates, authorized_for_submission=False,
                  independently_held_out=False, public_backbone_training_overlap=True,
                  fp32_candidate_preserved=True, gpu_run_seconds=terminal['elapsed_seconds'],
                  elapsed_seconds=time.monotonic()-start)
    result = helper['finite'](result)
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','summaries','gates','per_movie_delta','per_embryo_delta')},indent=2),flush=True)


if __name__ == '__main__':
    main()
