"""Synthetic-only scorer/transport checks; no competition truth is opened."""
import json
from pathlib import Path
import runpy
import warnings
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORING = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))


def payload():
    return dict(nodes={str(i):dict(node_id=i,t=i,z=3,y=40,x=40) for i in range(100)},
                edges=[dict(source_id=i,target_id=i+1) for i in range(99)])


def test_current_official_exact_graph_and_missed_edge():
    scorer=SCORING['load_scorer']()
    make=SCORING['prediction_graph']
    data=payload()
    graph,truth=make(data),make(data)
    er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
    assert (er.edge_tp,er.edge_fp,er.edge_fn)==(99,0,0)
    assert scorer.node_recall(graph,truth)==1.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert scorer.summarise([scorer.per_sample_metrics(er,100.,1.)])['score']==1.
    data['edges'].pop()
    graph,truth=make(data),make(payload())
    er=scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
    assert (er.edge_tp,er.edge_fp,er.edge_fn)==(98,0,1)


def test_full_receipt_rejects_smoke_or_changed_graph_before_truth(tmp_path):
    contract=ROOT/'.biohub/cache/public-d4-full-movie-v1-bundle/CONTRACT.json'
    sha=SCORING['sha']; stems=SCORING['STEMS']
    terminal=dict(status='complete_prelabel_predictions', contract_sha256=sha(contract),
        mode='full',elapsed_seconds=100.,inputs_unchanged=True,ground_truth_opened=False,
        independently_held_out=False,authorized_for_submission=False,movies={})
    for stem in stems:
        terminal['movies'][stem]={}
        for arm in ('original','corrected'):
            folder=tmp_path/f'{stem}-{arm}';folder.mkdir()
            path=folder/'prediction.json';path.write_text(json.dumps(payload()))
            terminal['movies'][stem][arm]=dict(frames=100,nodes=100,edges=99,prediction_sha256=sha(path))
    path=tmp_path/'result.json';path.write_text(json.dumps(terminal))
    result,graphs=SCORING['validate_predictions'](tmp_path,sha(path))
    assert len(graphs)==4
    terminal['mode']='smoke';path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError):SCORING['validate_predictions'](tmp_path,sha(path))
    terminal['mode']='full';path.write_text(json.dumps(terminal))
    (tmp_path/f'{stems[-1]}-corrected/prediction.json').write_text('{}')
    with pytest.raises(ValueError):SCORING['validate_predictions'](tmp_path,sha(path))
