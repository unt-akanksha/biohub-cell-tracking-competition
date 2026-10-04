"""Build a compact, hash-bound visual training bundle; no network/GPU calls."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.visual_correspondence_data_v1 import (
    EXCLUDED, MAX_CANDIDATES, VOXEL_UM, normalized_image, image_proposals,
    nearest_candidates, supervised_parent, patches_at, validate_roles,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    start = time.monotonic()
    archive_path = ROOT / '.biohub/cache/competition-real-localization-expanded-replay-v2/kernel-v3/biohub-real-localization-expanded-shards-v2.tar'
    output = ROOT / '.biohub/cache/visual-correspondence-ensemble-v1-bundle'
    output.mkdir(parents=True, exist_ok=False)
    data_root = output / 'data'; data_root.mkdir()
    inventory, counts = [], {}
    with tarfile.open(archive_path) as archive:
        prefix = 'competition_real_localization_expanded_shards_v2/'
        manifest_bytes = archive.extractfile(prefix + 'real_localization_shard_manifest.json').read()
        manifest = json.loads(manifest_bytes)
        if manifest['competition_test_data_read'] or manifest['status'] != 'complete':
            raise ValueError('Ineligible source archive')
        records = [r for r in manifest['files'] if r['role'] in ('optimization', 'selection') and r['stem'] not in EXCLUDED]
        validate_roles(records)
        for number, record in enumerate(records):
            raw = archive.extractfile(prefix + record['path']).read()
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('Source shard hash changed')
            with np.load(io.BytesIO(raw), allow_pickle=False) as a:
                volumes, nodes, edges = (a[k].copy() for k in ('volumes', 'nodes', 'edges'))
                if not np.allclose(a['voxel_um_pooled'], VOXEL_UM):
                    raise ValueError('Pooled scale changed')
            if len(edges) == 0:
                continue
            points = nodes[:, 1:4].copy(); points[:, 1:] /= 4
            images = [normalized_image(v) for v in volumes]
            proposals = [image_proposals(v) for v in images]
            # Precomputed image-only locations. GT is only consulted afterward.
            patch_requests, patch_ids, groups = [], {}, []
            def patch_id(t, coord):
                key = (t, *map(float, coord))
                if key not in patch_ids:
                    patch_ids[key] = len(patch_requests); patch_requests.append((t, coord.copy()))
                return patch_ids[key]
            incoming = {}
            for parent, child in edges:
                if int(child) in incoming:
                    raise ValueError('A child has more than one annotated parent')
                incoming[int(child)] = int(parent)
            ambiguous = 0
            for child, parent in incoming.items():
                pt, ct = int(nodes[parent, 0]), int(nodes[child, 0])
                if ct != pt + 1:
                    raise ValueError('Nonadjacent annotated edge')
                # Child coordinates are supervision anchors, not injected output nodes.
                child_coord = points[child]
                choice = nearest_candidates(proposals[pt], child_coord)
                candidate_coords = proposals[pt][choice]
                supervision = supervised_parent(candidate_coords, points[parent])
                if supervision is None:
                    ambiguous += 1; continue
                target, mask = supervision
                ids = np.full(MAX_CANDIDATES + 1, -1, np.int32)
                coords = np.zeros((MAX_CANDIDATES + 1, 3), np.float32)
                ids[0] = patch_id(ct, child_coord); coords[0] = child_coord * VOXEL_UM
                for j, coord in enumerate(candidate_coords):
                    ids[j+1] = patch_id(pt, coord); coords[j+1] = coord * VOXEL_UM
                loss_mask = np.zeros(MAX_CANDIDATES, bool); loss_mask[:len(mask)] = mask
                groups.append((ids, coords, loss_mask, target if target >= 0 else MAX_CANDIDATES))
            if not groups:
                continue
            patches = np.empty((len(patch_requests), 2, 15, 15, 15), np.float16)
            for t in range(3):
                rows = [i for i, request in enumerate(patch_requests) if request[0] == t]
                if rows:
                    patches[rows] = patches_at(images[t], np.stack([patch_requests[i][1] for i in rows]))
            ids, coords, masks, targets = (np.stack([g[i] for g in groups]) for i in range(4))
            if not np.isfinite(patches).all() or not np.isfinite(coords).all():
                raise ValueError('Nonfinite training packet')
            name = record['path'].replace('/', '__')
            destination = data_root / name
            np.savez_compressed(destination, patches=patches, ids=ids, coords=coords,
                                masks=masks, targets=targets.astype(np.int64))
            row = dict(path='data/' + name, sha256=sha(destination), bytes=destination.stat().st_size,
                       stem=record['stem'], embryo=record['embryo'], role=record['role'],
                       source_sha256=record['sha256'], groups=len(groups), patches=len(patches),
                       present=int(np.sum(targets != MAX_CANDIDATES)), null=int(np.sum(targets == MAX_CANDIDATES)),
                       ambiguous_excluded=ambiguous, multiple_safe_parents=int(np.sum(masks.sum(1) >= 2)))
            inventory.append(row)
            if (number + 1) % 50 == 0:
                print(json.dumps(dict(shards=number+1, groups=sum(r['groups'] for r in inventory), seconds=time.monotonic()-start)), flush=True)
    for embryo in ('44b6', '6bba'):
        for role in ('optimization', 'selection'):
            subset = [r for r in inventory if r['embryo'] == embryo and r['role'] == role]
            counts[embryo+'-'+role] = dict(movies=len({r['stem'] for r in subset}),
                **{k: sum(r[k] for r in subset) for k in ('groups', 'present', 'null', 'ambiguous_excluded', 'multiple_safe_parents')})
            if counts[embryo+'-'+role]['present'] < (50 if role == 'optimization' else 5):
                raise ValueError('Too few real positive correspondences; do not launch')
    files = {}
    for source, name in (
        ('research/visual_correspondence_data_v1.py', 'visual_correspondence_data_v1.py'),
        ('research/visual_correspondence_models_v1.py', 'visual_correspondence_models_v1.py'),
        ('scripts/run-visual-correspondence-ensemble-v1.py', 'run.py'),
    ):
        dest = output / name; dest.write_bytes((ROOT / source).read_bytes()); files[name] = sha(dest)
    contract = dict(run_id='visual-correspondence-ensemble-v1', status='prepared', records=inventory,
        files=files, counts=counts, source_archive_sha256=sha(archive_path),
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), excluded_stems=sorted(EXCLUDED),
        initialization='random; no public or Biohub checkpoint',
        families=['resnet3d', 'token_transformer3d'], source_embryos=['6bba', '44b6'],
        max_steps_per_member=5000, max_seconds_per_member=2400, total_watchdog_seconds=10080,
        batch_size=12, validation_every=250, seed=20260914,
        input='two image-derived 3D scales, 15-cube storage / 11-cube network; isotropic 1.625um',
        loss='known-child backward-parent softmax; ambiguous candidates masked; synthetic parent dropout 20%',
        training_coordinates='GT child queries with image-only parent proposals; jittered during training',
        no_pristine_project_holdout_claim=True, competition_test_data_read=False, sealed_audit_opened=False,
        requires_complete_movie_scoring_before_promotion=True, authorized_for_submission=False,
        dataset_bytes=sum(r['bytes'] for r in inventory), elapsed_seconds=time.monotonic()-start)
    (output / 'BUNDLE.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(dict(status='prepared', bundle=str(output), sha256=sha(output/'BUNDLE.json'),
                         counts=counts, bytes=contract['dataset_bytes'], seconds=contract['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
