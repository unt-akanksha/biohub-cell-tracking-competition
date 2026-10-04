import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/score-focus-owned-flow-full.py'))


def example():
    raw=json.loads((ROOT/'reports/experiments/focus-raw-learned-linker-v2-result.json').read_text())
    lookup={r['stem']:r for r in raw['per_movie']['raw_linker']}
    values=[lookup[s] for s in M['G']['STEMS']]
    rows={a:copy.deepcopy(values) for a in ('control','candidate')}
    summaries={a:copy.deepcopy(raw['summaries']['raw_linker']) for a in rows}
    embryos={a:copy.deepcopy(raw['by_embryo']['raw_linker']) for a in rows}
    return rows,summaries,embryos


def test_identical_output_is_not_a_pass():
    result=M['compare'](*example())
    assert not result['diagnostic_gate_passed']
    assert result['summary_deltas']['score']==0


@pytest.mark.parametrize('fault',['nodes','recall','nan','scope'])
def test_inconsistent_pair_rejected(fault):
    rows,summary,embryos=example()
    if fault=='nodes': rows['candidate'][0]['num_pred_nodes']+=1
    if fault=='recall': rows['candidate'][0]['node_recall']-=.01
    if fault=='nan': summary['candidate']['score']=float('nan')
    if fault=='scope': rows['candidate'][0]['stem']='44b6_unopened'
    with pytest.raises(ValueError): M['compare'](rows,summary,embryos)


def test_positive_aggregate_cannot_hide_embryo_regression():
    rows,summary,embryos=example()
    for r in rows['candidate']: r['adj_edge_jaccard']+=.001
    summary['candidate']['score']+=.001; summary['candidate']['edge_jaccard']+=.001
    embryos['candidate']['44b6']['score']-=.001
    embryos['candidate']['6bba']['score']+=.002
    result=M['compare'](rows,summary,embryos)
    assert not result['diagnostic_conditions']['neither_embryo_regresses']
    assert not result['diagnostic_gate_passed']


def test_undefined_division_values_serialize_as_null():
    value={'values':[float('nan'),float('inf'),1.], 'other':None}
    assert json.dumps(M['finite_json'](value),allow_nan=False)=='{"values": [null, null, 1.0], "other": null}'


def test_harvest_only_controller_cannot_push_or_submit(monkeypatch):
    path=ROOT/'scripts/run-focus-owned-flow-harvest.py'
    module=runpy.run_path(str(path)); assert module['validate']().is_file()
    source=path.read_text()
    assert "'push'" not in source and "'--accelerator'" not in source and "'submit'" not in source
    assert "open('x'" in source and 'helper.wait_complete' in source
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p: b'changed' if p.name=='biohub-focus-owned-flow-full-v1.ipynb' else read(p))
    with pytest.raises(ValueError,match='Frozen full diagnostic'): module['validate']()
