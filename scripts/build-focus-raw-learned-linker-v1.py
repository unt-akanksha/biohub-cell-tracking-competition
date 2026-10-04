"""Build the frozen diagnostic linker job after raw detections are recovered.

No launch or submission side effects. Raw terminal SHA is mandatory. Public
association weights overlap training movies: results cannot authorize promotion.
"""
import ast
import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / 'scripts/build-focus-bridge-official-paired-v2.py'))


def build(raw_terminal_sha256):
    if len(raw_terminal_sha256) != 64 or any(c not in '0123456789abcdef' for c in raw_terminal_sha256):
        raise ValueError('verified lowercase raw-terminal SHA-256 required')
    nb = BASE['build']()
    # Retain infrastructure and inference, not the base graph postprocessor
    # or bridge scorer. Inference owns the complete learned-linker graph.
    cells = []
    found = 0
    for cell in nb['cells']:
        source = ''.join(cell.get('source', []))
        if any(marker in source for marker in (
            'def filter_output_graph(', 'def apply_focus_bridge(',
            'Notebook-stage paired evaluation;')):
            continue
        if 'Prediction completed in' in source:
            found += 1
            anchor = 'start_time = time.time()'
            if source.count(anchor) != 1:
                raise ValueError('inference launch boundary changed')
            cache = (ROOT / 'research/focus_linker_cache.py').read_text()
            runtime = (ROOT / 'research/focus_linker_runtime.py').read_text()
            hook = '''
# Raw centroid adapter installed before CLI execution, after source patches.
from focus_linker_runtime import install as _install_focus_raw, solve_links_keep_nodes
import zarr as _focus_zarr
_focus_terminal = json.loads(Path(os.environ['FOCUS_RAW_TERMINAL']).read_text())
_install_focus_raw(globals(), os.environ['FOCUS_RAW_ROOT'], _focus_terminal,
    lambda path: tuple(_focus_zarr.open_group(str(path), mode='r')['0'].shape))
'''
            injection = f'''
import hashlib as _focus_hash
_raw_receipt_paths = [Path('/kaggle/input') / prefix / 'focus3d_raw_detections_terminal.json'
    for prefix in ('biohub-focus3d-raw-detections-v1',
                   'notebooks/indarkarhana/biohub-focus3d-raw-detections-v1',
                   'kernels/indarkarhana/biohub-focus3d-raw-detections-v1')]
_raw_receipts = [p for p in _raw_receipt_paths if p.is_file()
    and _focus_hash.sha256(p.read_bytes()).hexdigest() == {raw_terminal_sha256!r}]
if len(_raw_receipts) != 1:
    raise RuntimeError('Expected one hash-bound raw detection terminal')
_raw_receipt = _raw_receipts[0]
_raw_manifest = json.loads(_raw_receipt.read_text())
if set(r['stem'] for r in _raw_manifest['raw_detections']) != set(test_stems):
    raise RuntimeError('Raw detection movie coverage differs from frozen images')
os.environ['FOCUS_RAW_TERMINAL'] = str(_raw_receipt)
os.environ['FOCUS_RAW_ROOT'] = str(_raw_receipt.parent / 'raw_detections')
(REPO_DIR / 'scripts/focus_linker_cache.py').write_text({cache!r})
(REPO_DIR / 'scripts/focus_linker_runtime.py').write_text({runtime!r})
_focus_predictor_path = REPO_DIR / 'scripts/predict_unet_transformer.py'
_focus_predictor_text = _focus_predictor_path.read_text()
_focus_solver_anchor = 'graph = solver.solve(graph)'
if _focus_predictor_text.count(_focus_solver_anchor) != 1:
    raise RuntimeError('Predictor ILP solve boundary changed')
_focus_predictor_text = _focus_predictor_text.replace(
    _focus_solver_anchor, 'graph = solve_links_keep_nodes(graph, solver)')
_focus_main_anchor = 'if __name__ == "__main__":'
if _focus_predictor_text.count(_focus_main_anchor) != 1:
    raise RuntimeError('Predictor CLI entry point changed')
_focus_predictor_path.write_text(_focus_predictor_text.replace(
    _focus_main_anchor, {hook!r} + '\\n' + _focus_main_anchor))
'''
            source = source.replace(anchor, injection + '\n' + anchor)
        source = source.replace('focus-bridge-official-paired-v2', 'focus-raw-learned-linker-v1')
        cell['source'] = source.splitlines(keepends=True)
        cells.append(cell)
    if found != 1:
        raise ValueError('expected exactly one inference stage')
    # No GT scoring is built yet: recover full graphs and runtime first.
    terminal = '''
import hashlib
_raw_linked_dir = _prediction_dir_for_method(METHOD)
_raw_linked_paths = sorted(_raw_linked_dir.glob('*.geff'))
assert {p.stem for p in _raw_linked_paths} == set(test_stems)
_raw_linked_hashes = {}
for path in _raw_linked_paths:
    digest = hashlib.sha256()
    for part in sorted(path.rglob('*')):
        if part.is_file():
            digest.update(part.relative_to(path).as_posix().encode())
            digest.update(b'\\0'); digest.update(part.read_bytes()); digest.update(b'\\0')
    _raw_linked_hashes[path.stem] = digest.hexdigest()
Path('/kaggle/working/focus_raw_linked_graphs.json').write_text(json.dumps({
    'graph_root': str(_raw_linked_dir), 'graph_sha256': _raw_linked_hashes,
    'validation_scope': 'base-training diagnostic',
    'authorized_for_submission': False, 'authorized_for_production_promotion': False,
    'ground_truth_opened': False}, indent=2))
_LC_FINISHED = True
_LC_TIMER.cancel()
_lc_write_terminal('completed')
'''
    cells.append(BASE['BASE']['code_cell'](terminal))
    for cell in cells:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    nb['cells'] = cells
    nb['metadata']['codex'].update(run_id='focus-raw-learned-linker-v1',
        raw_terminal_sha256=raw_terminal_sha256, ground_truth_opened=False,
        authorized_for_production_promotion=False, launch_status='not_launched')
    return nb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-terminal-sha256', required=True)
    parser.add_argument('--raw-version', type=int, required=True)
    args = parser.parse_args()
    if args.raw_version < 1:
        raise ValueError('completed raw kernel version required')
    notebook = build(args.raw_terminal_sha256)
    target = ROOT / 'kaggle/biohub-focus-raw-learned-linker-v1'
    target.mkdir(parents=True, exist_ok=True)
    filename = target.name + '.ipynb'
    (target / filename).write_text(json.dumps(notebook))
    metadata = json.loads((BASE['TARGET'] / 'kernel-metadata.json').read_text())
    metadata.update(id='indarkarhana/' + target.name,
                    title='Biohub FOCUS Raw Learned Linker v1', code_file=filename,
                    kernel_sources=[f'indarkarhana/biohub-focus3d-raw-detections-v1/{args.raw_version}'])
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(target)
