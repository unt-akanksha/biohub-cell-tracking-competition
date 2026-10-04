"""Extend the successful frozen smoke to eight complete source movies only."""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/build-focus-source-probe.py'))
RUN = 'focus-source-cache-v1'
SLUG = 'biohub-' + RUN
SMOKE_SHA = '1ad40fc75748219cc9f6b6b5a1deb8f14cfd67dfd8242ad18dcc0135cc85a6e7'
SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def build(probe, split_bytes):
    if (probe['status'] != 'verified_six_frame_focus_source_probe' or probe['cpu_exact_replay'] is not True
            or probe['terminal']['status'] != 'completed' or probe['notebook_sha256'] != SMOKE_SHA
            or probe['authorized_for_submission'] is not False):
        raise ValueError('Verified exact GPU smoke required')
    if hashlib.sha256(split_bytes).hexdigest() != SPLIT_SHA:
        raise ValueError('Frozen source split required')
    fold = json.loads(split_bytes)['folds'][0]
    selection = fold['selection']
    if len(selection) != 8 or set(selection) & set(fold['train'] + fold['audit_order']):
        raise ValueError('Eight source-selection movies excluded from training required')
    parent = ROOT / 'kaggle/biohub-focus-source-probe-v1/biohub-focus-source-probe-v1.ipynb'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != SMOKE_SHA:
        raise ValueError('Frozen completed smoke notebook changed')
    nb = json.loads(parent.read_text())
    all_stems = M['STEMS'] + selection
    sources = [''.join(c['source']) for c in nb['cells']]
    sources[1] = sources[1].replace('focus-source-probe-v1', RUN).replace(
        'focus_source_probe_terminal.json', 'focus_source_cache_terminal.json')
    sources[1] = M['change'](sources[1], repr(M['STEMS']), repr(all_stems))
    sources[3] = M['change'](sources[3], f'valid_id = {M["STEMS"]!r}', f'valid_id = {all_stems!r}')
    sources[3] = M['change'](sources[3], 'MODE = "source_probe"', 'MODE = "source_cache"')
    sources[5] = M['change'](sources[5], 'volume = zarr_arr[:3]',
        f'limit = 3 if sample_id in {M["STEMS"]!r} else 100\n    volume = zarr_arr[:limit]\n    if list(volume.shape) != [limit, 64, 256, 256]:\n        raise ValueError("Incomplete source movie")')
    references = {r['stem']: r['sha256'] for r in probe['records']}
    if list(references) != M['STEMS']:
        raise ValueError('Both original smoke references required')
    terminal = f'''# Freeze every raw source output before any ground-truth scoring.
selection_stems = {selection!r}
smoke_references = {references!r}
_reference_slug = 'biohub-focus-source-probe-v1'
_reference_candidates = [Path('/kaggle/input') / p for p in (
    _reference_slug, 'notebooks/indarkarhana/' + _reference_slug,
    'kernels/indarkarhana/' + _reference_slug)]
_reference_roots = {{p.resolve() for p in _reference_candidates if (p / 'raw_detections').is_dir()}}
if len(_reference_roots) != 1:
    raise RuntimeError('Exactly one explicit completed smoke mount required')
_reference_root = _reference_roots.pop() / 'raw_detections'
records = []
for stem in valid_id:
    path = Path(predict_dir) / (stem + '.npz')
    record = json.loads(path.with_suffix('.json').read_text())
    expected_frames = 100 if stem in selection_stems else 3
    if (record['failed_frames'] != 0 or record['ground_truth_opened'] is not False
            or record['postprocessing_applied'] is not False
            or record['movie_shape'] != [expected_frames, 64, 256, 256]
            or len(record['frame_counts']) != expected_frames
            or sum(record['frame_counts']) != record['node_count']
            or record['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise RuntimeError('Incomplete or changed source detection output')
    with np.load(path, allow_pickle=False) as current:
        actual = current['coords'].copy()
    if stem in smoke_references:
        reference = _reference_root / (stem + '.npz')
        if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != smoke_references[stem]:
            raise RuntimeError('Frozen successful smoke artifact required')
        with np.load(reference, allow_pickle=False) as original:
            if not np.array_equal(actual, original['coords']):
                raise RuntimeError('Full source run failed exact small-probe replay')
    record['scope'] = 'source_selection' if stem in selection_stems else 'training_smoke_replay'
    records.append(record)
if Path('/kaggle/working/submission.csv').exists():
    raise RuntimeError('Source cache must not create a submission')
_FV_FINISHED = True
_fv_timer.cancel()
_fv_write('completed', records=records, selection_stems=selection_stems,
          exact_smoke_replay=True, ground_truth_opened=False, new_target_movies_opened=0,
          split_sha256={SPLIT_SHA!r}, model_sha256=_identity['model_sha256'])
'''
    nb['cells'] = [dict(cell_type='markdown', metadata={}, source=[
        '# FOCUS raw source cache\nEight fixed complete source movies plus six training replay frames. Attributed FOCUS author weights and qiweiyin runtime; no labels, postprocessing or submission.\n'])] + [M['cell'](s) for s in sources[1:7]] + [M['cell'](terminal)]
    nb['metadata']['codex'] = dict(run_id=RUN, source_stems=selection, replay_stems=M['STEMS'],
        split_sha256=SPLIT_SHA, smoke_notebook_sha256=SMOKE_SHA,
        model_sha256=probe['terminal']['model_sha256'], authorized_for_submission=False,
        pretraining_overlap_verified=False)
    meta = json.loads((parent.parent / 'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/' + SLUG, title=SLUG, code_file=SLUG + '.ipynb',
                kernel_sources=['indarkarhana/biohub-focus-source-probe-v1/1'])
    return nb, meta


if __name__ == '__main__':
    target = ROOT / 'kaggle' / SLUG
    if target.exists():
        raise ValueError('Refuse to overwrite staged source cache')
    probe = json.loads((ROOT / 'reports/experiments/focus-source-probe-v1-result.json').read_text())
    actual = runpy.run_path(str(ROOT / 'scripts/verify-focus-source-probe.py'))['verify'](
        ROOT / '.biohub/cache/kernel-outputs/focus-source-probe-v1')
    if actual != probe:
        raise ValueError('Verified smoke report changed')
    nb, meta = build(probe, (ROOT / 'research/independent_real_baseline_v1_split.json').read_bytes())
    target.mkdir()
    (target / (SLUG + '.ipynb')).write_text(json.dumps(nb))
    (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(target)
