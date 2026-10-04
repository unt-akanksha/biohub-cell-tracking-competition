"""Replay frozen caches, persist all proposed graphs, then score without GPU."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_edge_consensus import repair
from research.public_owned_flow_comparison import compare

RUN = 'public-owned-flow-repair-v1'
CACHE = ROOT / '.biohub/cache/kernel-outputs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    output = ROOT / f'.biohub/cache/{RUN}'
    if target.exists() or output.exists():
        raise ValueError('Refuse to overwrite a staged or completed diagnostic')
    contract_path = ROOT / 'research/public_owned_flow_repair_v1.json'
    contract = json.loads(contract_path.read_text())
    for name, expected in contract['pins'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    full = runpy.run_path(str(ROOT / 'scripts/score-focus-owned-flow-full.py'))
    raw = runpy.run_path(str(ROOT / 'scripts/score-focus-raw-linker.py'))
    replay = raw['REPLAY']
    prepared, _, _ = full['prepare'](
        CACHE / 'focus-owned-flow-full-v1/focus_owned_flow_full',
        ROOT / 'kaggle/biohub-focus-owned-flow-full-v1/biohub-focus-owned-flow-full-v1.ipynb')
    neural, _ = raw['validate'](CACHE / 'focus-raw-learned-linker-v2', CACHE / 'focus3d-raw-detections-v1')
    controls = replay['validate_arms'](CACHE / 'focus-bridge-cached-control-v2')
    if any(sorted(v) != contract['stems'] for v in (prepared, neural, controls)):
        raise ValueError('Exact frozen four-movie scope required')
    prior = json.loads((ROOT / 'reports/experiments/focus-edge-consensus-v1-result.json').read_text())
    import tracksdata as td
    from geff import GeffMetadata
    output.mkdir(parents=True)
    predictions, receipts = {}, {}
    for stem in contract['stems']:
        graph = td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0]
        external = dict(nodes={str(r['node_id']): {k: r[k] for k in ('node_id', 't', 'z', 'y', 'x')}
                               for r in graph.node_attrs().iter_rows(named=True)},
                        edges=[{k: int(r[k]) for k in ('source_id', 'target_id')}
                               for r in graph.edge_attrs().iter_rows(named=True)])
        base = controls[stem]['control']
        previous, _ = repair(base, neural[stem])
        if hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest() != prior['receipts'][stem]['prediction_sha256']:
            raise ValueError('Prior consensus replay changed')
        candidate, receipt = repair(base, external)
        predictions[stem] = dict(public_control=base, prior_consensus=previous, candidate=candidate)
        receipts[stem] = dict(repair=receipt, graphs={})
        for arm, payload in predictions[stem].items():
            if payload['nodes'] != base['nodes']:
                raise ValueError('Node mutation')
            path = output / f'{stem}-{arm}.json'
            path.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
            receipts[stem]['graphs'][arm] = dict(path=path.name, sha256=sha(path))
        print(json.dumps(dict(event='prelabel_prediction', stem=stem, **receipt)), flush=True)
    # Every arm for every movie is now persisted before opening ground truth.
    (output / 'prelabel_manifest.json').write_text(json.dumps(dict(
        contract_sha256=sha(contract_path), ground_truth_opened=False, receipts=receipts), indent=2))
    scorer = full['SCORER']['load_scorer'](ROOT / '.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    rows = {arm: [] for arm in ('public_control', 'prior_consensus', 'candidate')}
    for stem in contract['stems']:
        gt_path = ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train' / f'{stem}.geff'
        for arm in rows:
            saved = receipts[stem]['graphs'][arm]
            path = output / saved['path']
            if sha(path) != saved['sha256']:
                raise ValueError('Persisted prelabel graph changed')
            graph = replay['prediction_graph'](json.loads(path.read_text()))
            truth = td.graph.IndexedRXGraph.from_geff(str(gt_path))[0]
            er = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
            count = float(GeffMetadata.read(str(gt_path)).extra['estimated_number_of_nodes'])
            row = dict(scorer.per_sample_metrics(er, count, scorer.node_recall(graph, truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(dict(event='scored', arm=arm, **row)), flush=True)
    summaries = {arm: scorer.summarise(values) for arm, values in rows.items()}
    embryos = {arm: {e: scorer.summarise([r for r in values if r['embryo'] == e])
                     for e in ('44b6', '6bba')} for arm, values in rows.items()}
    result = dict(run_id=RUN, status='completed', contract=contract,
                  per_movie=rows, summaries=summaries, by_embryo=embryos, receipts=receipts,
                  comparison=compare(rows, summaries),
                  authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
                  source_sha256=sha(Path(__file__)), contract_sha256=sha(contract_path),
                  gpu_hours=0, new_target_movies_opened=0, authorized_for_submission=False,
                  validation_scope=contract['scope'])
    target.write_text(json.dumps(full['finite_json'](result), indent=2, allow_nan=False))
    print(json.dumps(result['comparison'], indent=2), flush=True)


if __name__ == '__main__':
    main()
