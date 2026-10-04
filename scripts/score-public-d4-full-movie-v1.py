"""Verify all frozen complete-movie predictions before opening local truth."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha, STEMS, csv_equivalent_graph
from research.public_d4_quality import compare


def load_scorer():
    folder = ROOT / '.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot'
    expected = {'metrics.py': 'cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444',
                'division_metrics.py': '0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9'}
    for name, digest in expected.items():
        if hashlib.sha256((folder/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Patched official scorer changed')
    name = '_d4_official_scorer'
    package = ModuleType(name); package.__path__ = [str(folder)]; sys.modules[name] = package
    return importlib.import_module(name+'.metrics')


def prediction_graph(payload):
    import polars as pl
    import tracksdata as td
    graph = td.graph.InMemoryGraph()
    for axis in ('z','y','x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.)
    ids = sorted(payload['nodes'], key=int)
    mapped = graph.bulk_add_nodes([{k: payload['nodes'][i][k] for k in ('t','z','y','x')} for i in ids])
    mapping = dict(zip(map(int, ids), mapped))
    graph.bulk_add_edges([dict(source_id=mapping[e['source_id']], target_id=mapping[e['target_id']])
                          for e in payload['edges']])
    return graph


def validate_predictions(folder, terminal_sha):
    contract_path = ROOT / '.biohub/cache/public-d4-full-movie-v1-bundle/CONTRACT.json'
    terminal = json.loads((folder/'result.json').read_text())
    if (sha(folder/'result.json') != terminal_sha or terminal['status'] != 'complete_prelabel_predictions'
            or terminal['contract_sha256'] != sha(contract_path) or terminal['mode'] != 'full'
            or not 0 < terminal['elapsed_seconds'] <= 3600 or not terminal['inputs_unchanged']
            or terminal['ground_truth_opened'] is not False or terminal['independently_held_out'] is not False
            or terminal['authorized_for_submission'] is not False
            or sorted(terminal['movies']) != list(STEMS)):
        raise ValueError('Complete same-contract label-free terminal required')
    prepared = {}
    for stem in STEMS:
        if set(terminal['movies'][stem]) != {'original','corrected'}:
            raise ValueError('Both arms required')
        prepared[stem] = {}
        for arm, record in terminal['movies'][stem].items():
            path = folder / f'{stem}-{arm}' / 'prediction.json'
            if record['frames'] != 100 or sha(path) != record['prediction_sha256']:
                raise ValueError('Incomplete or changed persisted graph')
            graph = json.loads(path.read_text())
            validated = csv_equivalent_graph({int(i):n for i,n in graph['nodes'].items()}, graph['edges'], 100)
            if (validated != graph or len(graph['nodes']) != record['nodes']
                    or len(graph['edges']) != record['edges']):
                raise ValueError('CSV-equivalent graph integrity failed')
            prepared[stem][arm] = graph
    return terminal, prepared


def finite(value):
    if isinstance(value, dict): return {k:finite(v) for k,v in value.items()}
    if isinstance(value, list): return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def main(args):
    if args.output.exists():
        raise ValueError('Refuse to overwrite a completed result')
    terminal, prepared = validate_predictions(args.predictions, args.terminal_sha256)
    scorer = load_scorer()
    import tracksdata as td
    from geff import GeffMetadata
    rows = {arm: [] for arm in ('original','corrected')}
    for stem in STEMS:
        truth_path = ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train' / (stem+'.geff')
        for arm in rows:
            # Official node matching mutates graphs: fresh instances per arm.
            truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            if set(truth.node_attrs()['t'].to_list()) != set(range(100)):
                raise ValueError('Ground truth is not the full 100-frame movie')
            graph = prediction_graph(prepared[stem][arm])
            er = scorer.evaluate(graph, truth, scale=(1.625,.40625,.40625), max_distance=7.)
            count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row = dict(scorer.per_sample_metrics(er,count,scorer.node_recall(graph,truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(finite(dict(event='scored', arm=arm, **row))), flush=True)
    summaries = {arm: scorer.summarise(values) for arm,values in rows.items()}
    embryos = {arm:{e:scorer.summarise([r for r in values if r['embryo']==e])
                    for e in ('44b6','6bba')} for arm,values in rows.items()}
    movies = {arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}
    comparison = compare(rows,summaries,embryos,movies)
    result = dict(run_id='public-d4-full-movie-v1', status='scored',
        source_sha256=sha(Path(__file__)), comparison_source_sha256=sha(ROOT/'research/public_d4_quality.py'),
        terminal_sha256=args.terminal_sha256, contract_sha256=terminal['contract_sha256'],
        per_movie=rows, summaries=summaries, by_embryo=embryos, per_movie_summaries=movies,
        comparison=comparison, independently_held_out=False, authorized_for_submission=False,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2')
    args.output.write_text(json.dumps(finite(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(finite(dict(summaries=summaries,comparison=comparison)),indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--terminal-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
