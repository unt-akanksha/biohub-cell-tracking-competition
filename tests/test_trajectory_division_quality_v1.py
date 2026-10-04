from pathlib import Path
import runpy

check = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                          'research/trajectory_division_quality_v1.py'))['check']


def example():
    stems = [f'{e}_{i}' for e in ('44b6','6bba') for i in range(4)]
    rows = {a:[dict(stem=s, edge_tp=11 if a=='repaired' else 10,
                   division_tp=1, division_fn=0, division_fp=0) for s in stems]
            for a in ('original','repaired')}
    summaries = {a:dict(score=.91 if a=='repaired' else .90,
                         edge_jaccard=.91 if a=='repaired' else .90) for a in rows}
    embryos = {a:{e:dict(score=summaries[a]['score']) for e in ('44b6','6bba')} for a in rows}
    movies = {a:{s:dict(score=summaries[a]['score']) for s in stems} for a in rows}
    return rows, summaries, embryos, movies, [stems[i] for i in (0,1,4,5)]


def test_requires_all_gates_not_just_pooled_gain():
    values = example(); assert check(*values)['diagnostic_gate_passed']
    values[3]['repaired']['44b6_0']['score'] = .89
    assert not check(*values)['diagnostic_gate_passed']


def test_division_damage_and_missing_positive_coverage_fail():
    values = example(); values[0]['repaired'][0]['division_fp'] = 1
    assert not check(*values)['diagnostic_gate_passed']
    values = example()
    for row in values[0]['original']:
        if row['stem'].startswith('44b6'):row['division_tp'] = 0
    assert not check(*values)['diagnostic_gate_passed']
