"""New immutable six-frame replay of the frozen raw FOCUS detector runtime."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUN = 'focus-source-probe-v1'
SLUG = 'biohub-' + RUN
STEMS = ['6bba_f1fde7e0', '6bba_23af9eeb']
PARENT = ROOT / 'kaggle/biohub-focus3d-raw-detections-v1/biohub-focus3d-raw-detections-v1.ipynb'
PARENT_SHA = '22a209eeb36b9ac9028c2b1e86ae8b32458518dfac674679e5c9ed801fb887a1'
AUDIT = runpy.run_path(str(ROOT / 'scripts/audit-focus-runtime-identity.py'))


def change(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Frozen source layout changed: ' + before[:60])
    return source.replace(before, after)


def cell(source):
    ast.parse(source)
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=source.splitlines(keepends=True))


def build(identity):
    if (identity['status'] != 'matched' or identity['model_sha256'] != AUDIT['EXPECTED_SHA']
            or identity['model_size'] != AUDIT['EXPECTED_SIZE'] or identity['weights_deserialized'] is not False):
        raise ValueError('Verified official nuclei checkpoint identity required')
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Frozen raw parent notebook changed')
    parent = json.loads(PARENT.read_text())
    sources = [''.join(c.get('source', [])) for c in parent['cells']]
    watchdog = sources[3].replace('focus3d-raw-detections-v1', RUN)
    watchdog = watchdog.replace('focus3d_raw_detections_terminal.json', 'focus_source_probe_terminal.json')
    watchdog = change(watchdog, "['44b6_81c256f0', '44b6_24264f12', '6bba_f1fde7e0', '6bba_23af9eeb']", repr(STEMS))
    watchdog = change(watchdog, 'Timer(10_800,', 'Timer(3300,')
    watchdog = change(watchdog, '"schema_version": 1,', '"schema_version": 1, "declared_budget_seconds": 3600,')
    setup = sources[4]
    start, end = setup.index('MODE = '), setup.index('# ---- FOCUS-3D runtime')
    setup = setup[:start] + f'MODE = "source_probe"\nvalid_id = {STEMS!r}\nvalid_dir = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"\n\n' + setup[end:]
    start, end = setup.index('from biohub_tracking.metrics import ('), setup.index('#--------------------------------------------')
    setup = setup[:start] + setup[end:]
    audit_source = (ROOT / 'scripts/audit-focus-runtime-identity.py').read_text()
    guard = f'''# Verify bytes before deserializing any model; no metric imported.
import hashlib
_audit_namespace = {{'__name__': '_focus_identity_guard'}}
exec(compile({audit_source!r}, 'identity_guard.py', 'exec'), _audit_namespace)
_identity = _audit_namespace['inspect_runtime'](FOCUS3D_ROOT)
_expected_identity = {identity!r}
if _identity['status'] != 'matched' or _identity['files'] != _expected_identity['files']:
    raise RuntimeError('Public runtime changed since CPU identity audit')
Path('/kaggle/working/verified_runtime_identity.json').write_text(json.dumps(_identity, indent=2))
if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('Exactly two T4 GPUs required')
'''
    detector = change(sources[6], 'volume = zarr_arr[:]', 'volume = zarr_arr[:3]')
    detector = change(detector, 'zarr_file = f"{valid_dir}/{sample_id}.zarr"',
                      'if sample_id not in valid_id:\n        raise ValueError("Movie outside frozen smoke scope")\n    zarr_file = f"{valid_dir}/{sample_id}.zarr"')
    terminal = '''# Require exact replay of the six already-cached raw frames.
_expected_raw = {
    '6bba_f1fde7e0': 'fcf6eb4a0586b1c77e5eaa9221543e1ec196ffcee091932a81dc1974fc40bc8d',
    '6bba_23af9eeb': '7d62559975797b50f9390ba68d2c76be358cd9c303b74459d1d871570111c7b2'}
records = []
for stem in valid_id:
    matches = list(Path('/kaggle/input').glob('**/raw_detections/' + stem + '.npz'))
    if len(matches) != 1 or hashlib.sha256(matches[0].read_bytes()).hexdigest() != _expected_raw[stem]:
        raise RuntimeError('Exact original raw reference required')
    with np.load(matches[0], allow_pickle=False) as original:
        expected = original['coords'][original['coords'][:, 0] < 3]
    path = Path(predict_dir) / (stem + '.npz')
    with np.load(path, allow_pickle=False) as current:
        actual = current['coords']
        if not np.array_equal(current['movie_shape'], [3, 64, 256, 256]) or not np.array_equal(actual, expected):
            raise RuntimeError('New detector differs from original raw six-frame replay')
    record = json.loads(path.with_suffix('.json').read_text())
    if (record['failed_frames'] != 0 or record['ground_truth_opened'] is not False
            or record['postprocessing_applied'] is not False
            or record['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise RuntimeError('Invalid smoke artifact')
    records.append(dict(stem=stem, frames=3, nodes=len(actual), sha256=record['sha256'], exact_replay=True))
if Path('/kaggle/working/submission.csv').exists():
    raise RuntimeError('Smoke must not create submission')
_FV_FINISHED = True
_fv_timer.cancel()
_fv_write('completed', records=records, exact_raw_replay=True, ground_truth_opened=False,
          model_sha256=_identity['model_sha256'])
'''
    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': parent['metadata']['kernelspec'],
        'language_info': parent['metadata']['language_info'],
        'codex': dict(run_id=RUN, parent_notebook_sha256=PARENT_SHA, frozen_stems=STEMS,
                      frames_per_movie=3, model_sha256=identity['model_sha256'], authorized_for_submission=False)},
        cells=[dict(cell_type='markdown', metadata={}, source=[
            '# FOCUS source-route smoke\nAttribution: qiweiyin/focus3d-nuclei-physical-pp-submit; FOCUS-3D BSD-3-Clause code, Apache-2.0 author weights. Six training frames only; no score or submission.\n']),
            *[cell(s) for s in (watchdog, sources[1], setup, guard, detector, sources[8], terminal)]])
    meta = json.loads((PARENT.parent / 'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/' + SLUG, title=SLUG, code_file=SLUG + '.ipynb',
                kernel_sources=['indarkarhana/biohub-focus3d-raw-detections-v1/1'])
    return nb, meta


if __name__ == '__main__':
    target = ROOT / 'kaggle' / SLUG
    if target.exists():
        raise ValueError('Refuse to overwrite frozen source smoke')
    identity = json.loads((ROOT / '.biohub/cache/kernel-outputs/focus-runtime-identity-v1/focus_runtime_identity.json').read_text())
    nb, meta = build(identity)
    target.mkdir()
    (target / (SLUG + '.ipynb')).write_text(json.dumps(nb))
    (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(target)
