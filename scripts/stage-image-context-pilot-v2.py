"""Build a self-contained, hash-pinned cloud bundle from verified local artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROOT / '.biohub/cache/image-context-pilot-v2-bundle'
    archive = ROOT / '.biohub/cache/image-context-pilot-v2-bundle.tar.gz'
    if output.exists() or archive.exists():
        raise ValueError('Preserve staged bundle')
    previous = ROOT / '.biohub/cache/graph-context-fresh-training-runtime-v4-remote'
    prior = json.loads((previous / 'runtime_manifest.json').read_text())
    terminal = json.loads((previous / 'pretraining_terminal.json').read_text())
    if terminal['competition_data_read'] is not False:
        raise ValueError('Warm start is not external-only')
    output.mkdir(parents=True)
    for name, row in prior['files'].items():
        if name.startswith('source__research__temporal_contrastive__'):
            relative = name.removeprefix('source__').replace('__', '/')
            # An inert package bootstrap avoids unrelated historical eager imports.
            if relative.endswith('/init/.py') or name.endswith('____init__.py'):
                continue
            destination = output / relative
        elif name.startswith('warm_start_') or name == 'pretraining_terminal.json':
            destination = output / name
        else:
            continue
        source = previous / name
        if sha(source) != row['sha256']:
            raise ValueError('Verified warm start/runtime changed')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for folder in ('research', 'research/temporal_contrastive'):
        (output / folder / '__init__.py').write_text('"""Isolated image-context runtime package."""\n')
    for name in ('research/image_division_context.py', 'research/image_context_quality.py',
                 'scripts/run-image-context-pilot-v2.py'):
        target = output / (Path(name).name if name.startswith('scripts/') else name)
        shutil.copyfile(ROOT / name, target)
    data = ROOT / '.biohub/cache/image-division-context-v2'
    dataset = json.loads((data / 'MANIFEST.json').read_text())
    if dataset['status'] != 'complete' or dataset['original_audit_shards_opened'] is not False:
        raise ValueError('Dataset contract failed')
    (output / 'data').mkdir()
    shutil.copyfile(data / 'MANIFEST.json', output / 'data/MANIFEST.json')
    for name, row in dataset['files'].items():
        for file_name, digest in ((name, row['sha256']), (row['inventory_path'], row['inventory_sha256'])):
            if sha(data / file_name) != digest:
                raise ValueError('Prepared data changed')
            shutil.copyfile(data / file_name, output / 'data' / file_name)
    manifest = dict(run_id='image-context-pilot-v2', parameter_count=74732308,
        files={str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        dataset_manifest_sha256=sha(data / 'MANIFEST.json'),
        design_sha256=sha(ROOT / 'reports/experiments/image-division-context-v2-design.md'),
        source_file=sha(ROOT / 'scripts/run-image-context-pilot-v2.py'),
        smoke_steps=20, pilot_steps_per_model=2000, model_count=2,
        gpu='Antelume only, initially idle, sequential models', authorized_for_submission=False)
    (output / 'BUNDLE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(output)).replace('\\', '/'))
    print(json.dumps(dict(status='staged', bundle_sha256=sha(output / 'BUNDLE.json'),
                         archive_sha256=sha(archive), bytes=archive.stat().st_size)))


if __name__ == '__main__':
    main()
