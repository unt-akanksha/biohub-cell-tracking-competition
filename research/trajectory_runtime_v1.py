"""Offline image-only input, bundle integrity and atomic CSV assembly."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

CSV_FIELDS = ('id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x',
              'source_id', 'target_id')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle, expected_sha):
    bundle = Path(bundle)
    if sha(bundle / 'CONTRACT.json') != expected_sha:
        raise ValueError('Runtime contract changed')
    contract = json.loads((bundle / 'CONTRACT.json').read_text())
    if contract['ground_truth_included'] or contract['public_prediction_tables_included']:
        raise ValueError('Runtime must contain models and code, not label/prediction tables')
    for name, digest in contract['bundle_sha256'].items():
        path = bundle / name
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or not path.resolve().is_relative_to(bundle.resolve())
                or sha(path) != digest):
            raise ValueError(f'Runtime input changed: {name}')
    return contract


def validate_movie_ids(movies):
    if (not movies or len(set(movies)) != len(movies)
            or any(not isinstance(m, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', m) for m in movies)):
        raise ValueError('Invalid/duplicate movie identifiers')


def image_metadata(path, *, normalize=False, load_image=False, downsample=(1, 4, 4)):
    if normalize or load_image or tuple(downsample) != (1, 4, 4):
        raise ValueError('Unexpected image-only loader request')
    path = Path(path)
    group = json.loads((path / 'zarr.json').read_text())['attributes']
    array = json.loads((path / '0/zarr.json').read_text())
    shape = array['shape']
    scale = group['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1:]
    if (len(shape) != 4 or shape[1:] != [64, 256, 256]
            or not isinstance(shape[0], int) or shape[0] < 8
            or scale != [1.625, .40625, .40625] or array['data_type'] != 'uint16'):
        raise ValueError('Unsupported movie geometry/dtype; no silent resampling')
    return SimpleNamespace(zarr_path=path, image_shape=(shape[0], 64, 64, 64),
                           scale=tuple(scale), quantiles=group['image_statistics']['quantiles'])


def validate_graph(graph, frames):
    nodes, edges = graph['nodes'], graph['edges']
    if not nodes or not edges:
        raise ValueError('Empty graph')
    for key, row in nodes.items():
        if (str(row['node_id']) != key or row['node_id'] < 0
                or any(type(row[k]) is not int for k in ('node_id', 't', 'z', 'y', 'x'))
                or any(not 0 <= row[k] < bound for k, bound in zip(('t','z','y','x'), (frames,64,256,256)))):
            raise ValueError('Invalid integer node identity, time or coordinates')
    if {n['t'] for n in nodes.values()} != set(range(frames)):
        raise ValueError('Missing movie frames')
    seen, parents, children = set(), Counter(), Counter()
    for edge in edges:
        a, b = edge['source_id'], edge['target_id']
        if (type(a) is not int or type(b) is not int or (a,b) in seen
                or str(a) not in nodes or str(b) not in nodes
                or nodes[str(b)]['t'] != nodes[str(a)]['t'] + 1):
            raise ValueError('Duplicate, dangling, noninteger or nonadjacent edge')
        seen.add((a,b)); parents[b] += 1; children[a] += 1
    if max(parents.values()) > 1 or max(children.values()) > 2:
        raise ValueError('Invalid lineage degree')


def graph_rows(movie, graph, frames):
    validate_movie_ids([movie])
    validate_graph(graph, frames)
    for ident in sorted(graph['nodes'], key=int):
        n = graph['nodes'][ident]
        yield (movie, 'node', n['node_id'], n['t'], n['z'], n['y'], n['x'], -1, -1)
    for e in sorted(graph['edges'], key=lambda e: (e['source_id'], e['target_id'])):
        yield (movie, 'edge', -1, -1, -1, -1, -1, e['source_id'], e['target_id'])


def assemble_csv(movie_outputs, frames_by_movie, output):
    """Publish only after every complete graph and the CSV row count pass."""
    output = Path(output)
    if output.exists() or output.with_suffix('.csv.partial').exists():
        raise ValueError('Preserve existing CSV outputs')
    if set(movie_outputs) != set(frames_by_movie):
        raise ValueError('Incomplete/extra movie output coverage')
    validate_movie_ids(list(movie_outputs))
    partial = output.with_suffix('.csv.partial')
    count, movie_counts = 0, {}
    with partial.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for movie in sorted(movie_outputs):
            graph = json.loads(Path(movie_outputs[movie]).read_text())
            start = count
            for row in graph_rows(movie, graph, frames_by_movie[movie]):
                writer.writerow((count,) + row); count += 1
            movie_counts[movie] = count-start
    with partial.open(newline='') as stream:
        reader = csv.reader(stream)
        if tuple(next(reader)) != CSV_FIELDS:
            raise ValueError('CSV header mismatch')
        observed = 0
        for index, row in enumerate(reader):
            if len(row) != len(CSV_FIELDS) or int(row[0]) != index:
                raise ValueError('Malformed/noncontiguous CSV rows')
            observed += 1
        if observed != count:
            raise ValueError('CSV row count mismatch')
    partial.rename(output)
    return dict(rows=count, movie_rows=movie_counts, sha256=sha(output))
