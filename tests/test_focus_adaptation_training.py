import pytest
from research.focus_adaptation_training import diagnostic_gate,SETTINGS


def row(loss,parent=80,absent=5):return dict(nll=loss,known_parent=100,known_absent=10,correct_parent=parent,correct_absent=absent)


def test_gate_requires_all_controls():
    assert diagnostic_gate(row(2),row(1.5),row(1.2))['passed']
    assert not diagnostic_gate(row(2),row(1.5),row(1.7))['passed']
    assert not diagnostic_gate(row(2),row(1.5),row(1.2,79))['passed']
    assert not diagnostic_gate(row(2),row(1.5),row(1.2,80,4))['passed']
    with pytest.raises(ValueError):diagnostic_gate(row(2),row(1.5),row(float('nan')))
    assert SETTINGS['steps']==800 and SETTINGS['smoke_steps']==4
