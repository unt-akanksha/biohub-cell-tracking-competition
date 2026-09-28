"""Fixed source-selection flow policy; never consumes target labels or counts."""
import hashlib
import json

SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
CACHE_NOTEBOOK_SHA = 'd65eac4ce6354829b13fd92176c3163b45a50efabe2c326f5b5322d7d73f322b'
FLOW_SHA = '3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788'
FLOW_TENSOR_SHA = 'e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779'
PROBE_MOTION_SHA = '8546bb5ee020a52e1074afe2e1a337997ec6323ffbdd286154159caf46755eba'
MODEL_SHA = 'b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a'
RUNTIME_SHA = '244277de3bee34ae4198da5d48deb6da2e747fb5c42bcb359d912815ef49a020'
PROBE_STEM = '6bba_23af9eeb'


def receipt(cache_payload, split_payload):
    if hashlib.sha256(split_payload).hexdigest() != SPLIT_SHA:
        raise ValueError('Exact frozen source split required')
    fold = json.loads(split_payload)['folds'][0]
    cache = json.loads(cache_payload)
    if (cache['status'] != 'verified_raw_focus_source_cache' or cache['run_id'] != 'focus-source-cache-v1'
            or cache['notebook_sha256'] != CACHE_NOTEBOOK_SHA or cache['model_sha256'] != MODEL_SHA
            or cache['split_sha256'] != SPLIT_SHA or cache['source_stems'] != fold['selection']
            or cache['complete_source_frames'] != 800 or cache['smoke_frames_replayed'] != 6
            or cache['ground_truth_opened'] is not False or cache['authorized_for_submission'] is not False
            or cache['terminal']['status'] != 'completed' or cache['terminal']['exact_smoke_replay'] is not True):
        raise ValueError('Verified complete source cache required')
    if (len(fold['selection']) != 8 or set(fold['selection']) & set(fold['train'] + fold['audit_order'])
            or PROBE_STEM not in fold['train']):
        raise ValueError('Source-only held-out-flow selection and training-only replay required')
    expected = ['6bba_f1fde7e0', PROBE_STEM] + fold['selection']
    if [r['stem'] for r in cache['records']] != expected:
        raise ValueError('Exact raw cache movie order required')
    for r in cache['records']:
        source = r['stem'] in fold['selection']
        if r['frames'] != (100 if source else 3) or r['scope'] != ('source_selection' if source else 'training_smoke_replay'):
            raise ValueError('Raw cache frame scope changed')
    return dict(version=1, run_id='focus-source-flow-v1', source_stems=fold['selection'],
        inference_stems=[PROBE_STEM] + fold['selection'], split_sha256=SPLIT_SHA,
        cache_report_sha256=hashlib.sha256(cache_payload).hexdigest(),
        raw_terminal_sha256=cache['terminal_sha256'], raw_records=cache['records'],
        runtime_identity_sha256=RUNTIME_SHA, model_sha256=MODEL_SHA,
        flow_checkpoint_sha256=FLOW_SHA, flow_tensor_sha256=FLOW_TENSOR_SHA,
        probe_motion_sha256=PROBE_MOTION_SHA, probe_stem=PROBE_STEM,
        flow_views=1, null_logit=-4.5, posterior_threshold=.5, max_parents=1, max_children=2,
        coordinates='exact unpruned raw FOCUS centroids', detector_inference=False,
        public_linker_loaded=False, ground_truth_opened=False, new_target_movies_opened=0,
        pretraining_overlap_verified=False, authorized_for_submission=False)


def verify_raw_cache(folder, policy):
    import numpy as np
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(folder / 'focus_source_cache_terminal.json') != policy['raw_terminal_sha256']:
        raise ValueError('Actual raw terminal changed')
    if sha(folder / 'verified_runtime_identity.json') != policy['runtime_identity_sha256']:
        raise ValueError('Actual detector runtime identity changed')
    records = {}
    for record in policy['raw_records']:
        path = folder / 'raw_detections' / (record['stem'] + '.npz')
        if sha(path) != record['sha256']:
            raise ValueError('Raw centroid bytes changed')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'coords', 'movie_shape', 'scale_um'}:
                raise ValueError('Raw schema changed')
            coords = data['coords'].copy()
            shape = [record['frames'], 64, 256, 256]
            if (coords.shape != (record['nodes'], 4) or coords.dtype != np.float32
                    or not np.array_equal(data['movie_shape'], shape)
                    or not np.array_equal(data['scale_um'], [1.625, .40625, .40625])
                    or not np.isfinite(coords).all() or np.any(coords < 0)
                    or np.any(coords >= np.asarray(shape))
                    or np.any(coords[:, 0] != np.floor(coords[:, 0]))):
                raise ValueError('Invalid raw centroid coordinates or geometry')
        records[record['stem']] = dict(record, frame_counts=[int(np.sum(coords[:, 0] == t)) for t in range(record['frames'])])
    return records
