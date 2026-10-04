from pathlib import Path
import copy
import pytest
from research.public_d4_full_movie import (
    public_config, csv_equivalent_graph, adapt_predictor_logs, original_postprocess)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_public_config_ignores_host_environment(monkeypatch):
    monkeypatch.setenv('BIOHUB_DET_THRESHOLD', '0.1')
    config = public_config(ROOT / '.biohub/cache/public-frontier-20260910-2132/lf-dctta/biohub-lf-dctta.ipynb')
    assert len(config['globals']) == 82
    assert config['globals']['DET_THRESHOLD'] == .965
    assert config['globals']['DEEPCENTER_SAFE_DIV_THRESHOLD'] == .25
    assert config['globals']['OUTPUT_LINEFIT_WEIGHT'] == .8
    assert config['environment']['BIOHUB_SECONDARY_LOW_MARGIN_MAX'] == '0.35'


def test_path_and_original_deepcenter_adapters_are_exact():
    root = ROOT / '.biohub/cache/public-d4-correction-v1'
    source = (root / 'public-predictor-original.py').read_text()
    adapted = adapt_predictor_logs(source)
    assert adapted.replace('_RUN_OUTPUT', 'Path("/kaggle/working")') == source
    fixed = (root / 'public-postprocess-d4-corrected.py').read_text()
    original = original_postprocess(fixed)
    assert original != fixed
    with pytest.raises(ValueError): original_postprocess(original)


def test_csv_roundtrip_and_lineage_rejections():
    nodes = {i:dict(node_id=i, t=i, z=2.5, y=4.6, x=3.) for i in range(3)}
    edges = [dict(source_id=i, target_id=i+1) for i in range(2)]
    result = csv_equivalent_graph(nodes, edges, 3)
    assert result['nodes']['0']['z'] == 2
    assert result['nodes']['0']['y'] == 5
    for bad in (edges+[edges[0]], [dict(source_id=0,target_id=2)], []):
        with pytest.raises(ValueError): csv_equivalent_graph(nodes,bad,3)
    invalid = copy.deepcopy(nodes); invalid[0]['x'] = float('nan')
    with pytest.raises(ValueError): csv_equivalent_graph(invalid,edges,3)
    invalid[0]['x'] = 256
    with pytest.raises(ValueError): csv_equivalent_graph(invalid,edges,3)
