"""Replay our CSV through pinned official converters, without labels or GPU."""
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import assemble_csv, sha


def main():
    import polars as pl
    import tracksdata as td
    started=time.monotonic()
    out=ROOT/'.biohub/cache/trajectory-portable-csv-verification-v1'
    out.mkdir(exist_ok=False)
    cached=ROOT/'.biohub/cache/trajectory-division-full-v1-output'
    original=json.loads((cached/'result.json').read_text())
    if sha(cached/'result.json')!='40c35432a298d22f347cbdd111b1cd6d21ea3965acc35b4fc63369d3147544b1':
        raise ValueError('Frozen terminal changed')
    paths={m:cached/f'{m}-original/repaired-prediction.json' for m in original['movies']}
    for m,p in paths.items():
        if sha(p)!=original['movies'][m]['original']['repaired_sha256']:
            raise ValueError('Frozen graph changed')
    result=assemble_csv(paths,{m:100 for m in paths},out/'validation-predictions.csv')
    namespace=dict(pl=pl,td=td)
    sources={}
    for name,filename in [('build_graph_from_rows','csv_to_geffs.py'),('graph_to_rows','geffs_to_csv.py')]:
        path=ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/scripts'/filename
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        text='from __future__ import annotations\n'+ast.unparse(ast.Module(body=[node],type_ignores=[]))
        exec(compile(text,filename,'exec'),namespace)
        sources[filename]=sha(path)
    table=pl.read_csv(out/'validation-predictions.csv')
    for movie in paths:
        rows=table.filter(pl.col('dataset')==movie)
        graph=namespace['build_graph_from_rows'](rows.filter(pl.col('row_type')=='node'),rows.filter(pl.col('row_type')=='edge'))
        rebuilt=namespace['graph_to_rows'](graph,movie)
        def signatures(frame):
            node_rows=frame.filter(pl.col('row_type')=='node')
            coords={r['node_id']:tuple(r[k] for k in ('t','z','y','x')) for r in node_rows.iter_rows(named=True)}
            edges=Counter((coords[r['source_id']],coords[r['target_id']])
                          for r in frame.filter(pl.col('row_type')=='edge').iter_rows(named=True))
            return Counter(coords.values()),edges
        if signatures(rows)!=signatures(rebuilt):
            raise ValueError('Official CSV roundtrip changed graph')
    receipt=dict(status='verified',movies=list(paths),csv=result,official_converter_sha256=sources,
                 node_and_edge_coordinate_multisets_identical=True,ground_truth_opened=False,
                 gpu_hours=0,elapsed_seconds=time.monotonic()-started,submission_performed=False)
    (ROOT/'reports/experiments/trajectory-portable-csv-v1-verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
