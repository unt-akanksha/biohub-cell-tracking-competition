"""Hash-bound, frozen source-only encoder features; no classifier fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.native_correspondence_models_v2 import VisualCorrespondence, prepare_patches
from research.native_division_features_v3 import joint_features, patch_statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_guard():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(line.strip() and int(line.strip()) != os.getpid() for line in result.splitlines()):
        raise RuntimeError('Another process owns GPU memory; leave it untouched')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-proof', type=Path)
    args = parser.parse_args()
    contract_path = ROOT / 'CONTRACT.json'
    contract = json.loads(contract_path.read_text())
    for item in contract['files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Code contract mismatch')
    data_path = args.data / 'DATA.json'
    if sha(data_path) != contract['data_sha256']:
        raise ValueError('Division data changed')
    data = json.loads(data_path.read_text())
    if not args.smoke:
        proof = json.loads(args.smoke_proof.read_text())
        if proof['status'] != 'feature_smoke_passed' or proof['contract_sha256'] != sha(contract_path):
            raise ValueError('Full extraction requires matching functionality smoke')
    gpu_guard()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    records = data['records']
    if args.smoke:
        records = [next(r for r in records if r['embryo'] == e and r['role'] == 'optimization' and r['positive'] > 0)
                   for e in ('44b6', '6bba')]
    patches = []; triples = []; labels = []; coords = []; rows = []; offset = 0
    for record in records:
        path = args.data / record['path']
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Packet provenance mismatch')
        with np.load(path, allow_pickle=False) as packet:
            p = packet['patches']; t = packet['triples']; y = packet['labels']
            if p.shape[1:] != (3, 15, 15, 15) or t.shape != (len(y), 3) or t.min() < 0 or t.max() >= len(p):
                raise ValueError('Invalid image-triplet geometry')
            if int(y.sum()) != record['positive'] or len(y) != record['triples']:
                raise ValueError('Triplet counts changed')
            patches.append(p); triples.append(t + offset); labels.append(y); coords.append(packet['coords'])
            rows.extend([dict(stem=record['stem'], embryo=record['embryo'], role=record['role'], transition=record['transition'])] * len(y))
            offset += len(p)
    stored = torch.from_numpy(np.concatenate(patches))
    triples = torch.from_numpy(np.concatenate(triples)).long()
    coords = torch.from_numpy(np.concatenate(coords)).float()
    statistics = patch_statistics(stored)
    row_path = args.output / 'ROWS.json'
    save(row_path, rows)
    results = []
    with torch.inference_mode():
        for source in ('44b6', '6bba'):
            gpu_guard()
            encoders = [e for e in data['encoders'] if e['embryo'] == source]
            encoders.sort(key=lambda e: ('resnet3d', 'token_transformer3d').index(e['family']))
            embeddings = []
            for member in encoders:
                if sha(Path(member['path'])) != member['sha256']:
                    raise ValueError('Source encoder hash mismatch')
                model = VisualCorrespondence(member['family'], input_channels=3).cuda().eval()
                model.load_state_dict(torch.load(member['path'], map_location='cuda', weights_only=True)['state_dict'])
                outputs = []
                for start in range(0, len(stored), 64):
                    if time.monotonic() - started > contract['feature_max_seconds']:
                        raise TimeoutError('Feature budget exhausted')
                    batch = stored[start:start + 64].cuda()
                    valid = torch.ones((len(batch), 1), dtype=torch.bool, device='cuda')
                    crop, _ = prepare_patches(batch[:, None], torch.zeros(len(batch), 1, 3, device='cuda'), valid)
                    with torch.autocast('cuda', dtype=torch.float16):
                        encoded = model.encoder(crop[:, 0])
                    if not torch.isfinite(encoded).all():
                        raise ValueError('Nonfinite native encoder features')
                    if args.smoke and start == 0:
                        model.load_state_dict(torch.load(member['path'], map_location='cuda', weights_only=True)['state_dict'])
                        with torch.autocast('cuda', dtype=torch.float16):
                            reloaded = model.encoder(crop[:, 0])
                        torch.testing.assert_close(encoded, reloaded, rtol=0, atol=0)
                    outputs.append(encoded.half().cpu())
                embeddings.append(torch.cat(outputs))
                del model
                torch.cuda.empty_cache()
            features = joint_features(embeddings, statistics, triples, coords)
            swapped = joint_features(embeddings, statistics, triples[:, [0, 2, 1]], coords[:, [0, 2, 1]])
            torch.testing.assert_close(features, swapped, rtol=0, atol=0)
            path = args.output / (source + '.npz')
            np.savez_compressed(path, features=features.numpy(), labels=np.concatenate(labels),
                                embryo=np.array([r['embryo'] for r in rows]), role=np.array([r['role'] for r in rows]),
                                stem=np.array([r['stem'] for r in rows]))
            results.append(dict(source=source, path=path.name, sha256=sha(path), bytes=path.stat().st_size,
                                rows=len(features), columns=features.shape[1], encoders=encoders))
            print(json.dumps(dict(source=source, features=list(features.shape), seconds=time.monotonic() - started)), flush=True)
    result = dict(status='feature_smoke_passed' if args.smoke else 'features_complete', seconds=time.monotonic() - started,
                  contract_sha256=sha(contract_path), data_sha256=sha(data_path), rows_sha256=sha(row_path), members=results,
                  patches=len(stored), selection_labels_used_for_fitting=False, target_pilot_labels_used=False,
                  authorized_for_submission=False)
    save(args.output / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
