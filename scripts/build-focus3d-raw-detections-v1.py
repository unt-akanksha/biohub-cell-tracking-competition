"""Prepare reusable label-blind, pre-postprocessing FOCUS detections.

No launch side effects. Uses the previously frozen four bridge images and
unchanged detector parameters, but no physical linker, filtering or smoothing.
"""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / 'scripts/build-focus3d-bridge-proposals-v1.py'))

SAVE_RAW = '''def _physical_pp_and_save(out_node, out_edge, meta, sample_id, gpu_id, t_video0):
    # Historical function name retained only for the existing worker dispatch.
    import hashlib
    coords = np.asarray(out_node, dtype=np.float32).reshape(-1, 4)
    shape = tuple(meta['movie_shape'])
    if out_edge or not np.isfinite(coords).all():
        raise RuntimeError('Raw detection contract violated')
    if np.any(coords < 0) or np.any(coords >= np.asarray(shape)):
        raise RuntimeError('Raw centroid outside image bounds')
    if np.any(coords[:, 0] != np.floor(coords[:, 0])):
        raise RuntimeError('Fractional detection frame')
    target = Path(predict_dir) / f'{sample_id}.npz'
    np.savez_compressed(target, coords=coords, movie_shape=np.asarray(shape),
                        scale_um=np.asarray(meta['scale_um']))
    manifest = {
        'stem': sample_id, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'movie_shape': list(shape), 'node_count': len(coords),
        'frame_counts': [int(np.sum(coords[:, 0] == t)) for t in range(shape[0])],
        'coordinate_source': 'raw instance centroids', 'postprocessing_applied': False,
        'failed_frames': 0, 'ground_truth_opened': False,
    }
    target.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print('Saved raw detection checkpoint', manifest, flush=True)
    return sample_id


'''

TERMINAL = '''# Raw detections only: no labels, graph postprocessing or CSV.
import hashlib
records = []
for stem in valid_id:
    target = Path(predict_dir) / f'{stem}.npz'
    record = json.loads(target.with_suffix('.json').read_text())
    assert record['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert sum(record['frame_counts']) == record['node_count']
    assert record['failed_frames'] == 0 and not record['postprocessing_applied']
    records.append(record)
assert not Path('/kaggle/working/submission.csv').exists()
_FV_FINISHED = True
_fv_timer.cancel()
_fv_write('completed', raw_detections=records, complete_movie_count=len(records),
          ground_truth_opened=False, postprocessing_applied=False)
'''


def build():
    notebook = BASE['build_notebook']()
    found = {'detector': 0, 'worker': 0, 'terminal': 0, 'watchdog': 0}
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if 'def predict_one_focus3d(' in source:
            found['detector'] += 1
            source = source.replace('return volume, meta',
                                    "meta['movie_shape'] = list(volume.shape)\n    return volume, meta")
            source = source.replace('return out_node, out_edge',
                                    "if n_failed:\n        raise RuntimeError(f'{n_failed} FOCUS frames failed')\n    return out_node, out_edge")
        elif 'def _physical_pp_and_save(' in source:
            found['worker'] += 1
            start = source.index('def _physical_pp_and_save(')
            end = source.index('def run_worker(', start)
            source = source[:start] + SAVE_RAW + source[end:]
            source = source.replace('/kaggle/working/my_predict', '/kaggle/working/raw_detections')
        elif '# Bind proposal graphs' in source:
            found['terminal'] += 1
            source = TERMINAL
        elif '_fv_prediction_paths = ' in source:
            source = source.replace('f"{stem}.geff"', 'f"{stem}.npz"')
            source = source.replace('prediction graphs', 'raw detection checkpoints')
        elif '_FV_RUN_ID = ' in source:
            found['watchdog'] += 1
            source = source.replace('focus3d-bridge-proposals-v1', 'focus3d-raw-detections-v1')
            source = source.replace('focus3d_bridge_proposals_terminal.json', 'focus3d_raw_detections_terminal.json')
        # Physical PP implementation is unnecessary and must not be executable.
        if 'def postprocess_tracks(' in source:
            source = '# No physical postprocessing in the raw detector experiment.\n'
        cell['source'] = source.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            ast.parse(source)
    if found != dict.fromkeys(found, 1):
        raise RuntimeError(f'Source layout changed: {found}')
    notebook['metadata']['codex'].update({
        'run_id': 'focus3d-raw-detections-v1',
        'coordinate_source': 'raw instance centroids',
        'postprocessing_applied': False,
        'authorized_for_submission': False,
        'launch_status': 'not_launched',
    })
    return notebook


if __name__ == '__main__':
    target = ROOT / 'kaggle/biohub-focus3d-raw-detections-v1'
    target.mkdir(parents=True, exist_ok=True)
    filename = target.name + '.ipynb'
    (target / filename).write_text(json.dumps(build()))
    metadata = json.loads((BASE['TARGET_DIR'] / 'kernel-metadata.json').read_text())
    metadata.update(id='indarkarhana/' + target.name,
                    title='Biohub FOCUS3D Raw Detections v1', code_file=filename)
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(target)
