import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_persists_all_arms_before_opening_labels():
    module = runpy.run_path(str(ROOT/'scripts/build-focus-bridge-official-paired-v2.py'))
    notebook = module['build']()
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    for source in sources:
        ast.parse(source)
    text = '\n'.join(sources)
    assert 'TEST_DIR = COMP_DIR / "test"' not in text
    assert 'control_validation.csv' in text
    assert text.count('require_tracks=True') == 1
    assert text.index('prelabel_graph_manifest.json') < text.index('require_tracks=True')
    assert 'official.evaluate(graph, ds.tracks' in text
    assert 'def compute_division_confusion' not in text
    assert 'kaggle competitions submit' not in text
    assert 'Timer(10200,' in text
    contract = json.loads((ROOT/'research/focus_bridge_official_paired_v2_contract.json').read_text())
    assert contract['validation_stems'] == module['STEMS']
