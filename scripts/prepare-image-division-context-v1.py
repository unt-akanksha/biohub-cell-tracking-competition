"""Prepare independent per-example image context; never open v3 audit shards."""
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from research.image_division_context import image_context


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    dest = ROOT / '.biohub/cache/image-division-context-v1'
    if dest.exists():
        raise ValueError('Preserve staged or completed dataset')
    split_path = ROOT / 'research/graph_context_fresh_split_v3.json'
    if sha(split_path) != '8d53a55217be88efc895ae9bf0bf378a3a4acb6c42437836a342d888cc9296da':
        raise ValueError('Split changed')
    roles = {r['stem']: r['role'] for r in json.loads(split_path.read_text())['stem_roles']}
    root = ROOT / '.biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1'
    manifest_path = root / 'graph_context_relational_patch_manifest.json'
    if sha(manifest_path) != '2e5c4c46b11ff1480194c29e88c0e16b0ebf2f0fd65b2fc29548d26fee599bd9':
        raise ValueError('Source manifest changed')
    source = json.loads(manifest_path.read_text())
    pins = {name: sha(ROOT / name) for name in ('research/image_division_context.py',
        'scripts/prepare-image-division-context-v1.py',
        'reports/experiments/image-division-context-v1-design.md')}
    dest.mkdir(parents=True)
    files, originals = {}, []
    for embryo in ('44b6', '6bba'):
        for role in ('optimization', 'selection'):
            grouped = defaultdict(list)
            inventory = []
            records = [r for r in source['records'] if r['embryo'] == embryo and roles[r['stem']] == role]
            for index, row in enumerate(records):
                path = root / row['path']
                if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                    raise ValueError('Source shard changed')
                with np.load(path, allow_pickle=False) as a:
                    patches = a['relational_patches']
                    old_context = a['graph_context_features']
                    old_mask = a['graph_context_mask']
                    if patches.shape != (row['rows'], 3, 3, 17, 17, 17) or not old_mask[:, :3].all():
                        raise ValueError('Source patch/anchor contract changed')
                    contexts, masks = [], []
                    for patch, anchors in zip(patches, old_context[:, :3]):
                        context, mask = image_context(patch[0], anchors)
                        contexts.append(context);masks.append(mask)
                    grouped['patches'].append(patches)
                    grouped['geometry'].append(a['geometry_features'])
                    grouped['context'].append(np.stack(contexts).astype(np.float16))
                    grouped['mask'].append(np.stack(masks))
                    grouped['targets'].append(a['division_recovery_target'])
                    grouped['weights'].append(a['label_weight'])
                    grouped['eligible'].append(a['inference_geometry_eligible'])
                    inventory.extend(dict(stem=row['stem'], embryo=embryo, role=role,
                        source_path=row['path'], source_row=i) for i in range(row['rows']))
                originals.append(dict(path=row['path'], sha256=row['sha256'], rows=row['rows'], new_role=role))
                if (index + 1) % 100 == 0:
                    print(json.dumps(dict(event='features', embryo=embryo, role=role,
                        shards=index + 1, total_shards=len(records))), flush=True)
            arrays = {k: np.concatenate(v) for k, v in grouped.items()}
            if any(len(a) != len(inventory) for a in arrays.values()):
                raise ValueError('Misaligned prepared arrays')
            path = dest / f'{embryo}-{role}.npz'
            np.savez_compressed(path, **arrays)
            inventory_path = dest / f'{embryo}-{role}-inventory.json'
            inventory_path.write_text(json.dumps(inventory, indent=2))
            positives = arrays['targets'] > .5
            eligible = arrays['eligible'].astype(bool)
            row = dict(bytes=path.stat().st_size, sha256=sha(path), rows=len(inventory),
                movies=len({x['stem'] for x in inventory}), positive=int(positives.sum()),
                eligible_positive=int((positives & eligible).sum()),
                eligible_negative=int((~positives & eligible).sum()),
                image_context_mean_valid_tokens=float(arrays['mask'].sum(axis=1).mean()),
                inventory_path=inventory_path.name, inventory_sha256=sha(inventory_path))
            files[path.name] = row
            print(json.dumps(dict(event='packet_complete', packet=path.name, **row)), flush=True)
    if any(sha(ROOT / n) != digest for n, digest in pins.items()):
        raise ValueError('Frozen feature contract changed')
    manifest = dict(run_id='image-division-context-v1', status='complete', files=files,
        source_pins=pins, source_records=originals, source_manifest_sha256=sha(manifest_path),
        roles_sha256=sha(split_path), v3_audit_shards_opened=False, gpu_hours=0,
        neighborhood_gt_nodes_used=False, supervised_anchor_coordinates_used=True,
        source_labels_copied_without_relabeling=True, cross_example_normalization=False,
        authorized_for_submission=False, elapsed_seconds=time.monotonic()-started)
    (dest / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='complete', files=files, elapsed_seconds=manifest['elapsed_seconds'],
                          manifest_sha256=sha(dest / 'MANIFEST.json'))), flush=True)


if __name__ == '__main__':
    main()
