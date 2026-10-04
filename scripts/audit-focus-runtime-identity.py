"""Read-only, stdlib-only public runtime identity check. Never deserialize weights."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

EXPECTED_SHA = 'b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a'
EXPECTED_SIZE = 4468804784
REVISION = '115258efcc9ee44e69db3902bce2511d0ae24e2f'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_runtime(root, expected_sha=EXPECTED_SHA, expected_size=EXPECTED_SIZE):
    weights = sorted(root.rglob('model_final_nuclei.pth'))
    if len(weights) != 1:
        raise ValueError('Exactly one nuclei checkpoint required')
    checkpoint = weights[0]
    actual_size = checkpoint.stat().st_size
    actual_sha = sha(checkpoint)
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        row = dict(path=relative, size=path.stat().st_size)
        if path == checkpoint:
            row['sha256'] = actual_sha
        elif path.suffix.lower() in ('.py', '.yaml', '.yml', '.json', '.toml', '.txt', '.md') or path.name.lower().startswith(('license', 'notice')):
            row['sha256'] = sha(path)
        files.append(row)
    return dict(status='matched' if (actual_size == expected_size and actual_sha == expected_sha) else 'mismatch',
                model_sha256=actual_sha, model_size=actual_size,
                expected_sha256=expected_sha, expected_size=expected_size,
                official_revision=REVISION, files=files,
                weights_deserialized=False, competition_data_read=False,
                authorized_for_submission=False, gpu_hours=0)


def main():
    started = time.monotonic()
    output = Path('/kaggle/working/focus_runtime_identity.json')
    if output.exists():
        raise ValueError('Refuse to overwrite identity receipt')
    def timeout():
        output.write_text(json.dumps(dict(status='timeout', declared_budget_seconds=900)))
        os._exit(124)
    timer = threading.Timer(900, timeout)
    timer.daemon = True
    timer.start()
    try:
        candidates = [Path('/kaggle/input') / p for p in (
            'datasets/qiweiyin/focus3d-nuclei-runtime',
            'qiweiyin/focus3d-nuclei-runtime', 'focus3d-nuclei-runtime')]
        roots = {p.resolve() for p in candidates if p.is_dir()}
        if len(roots) != 1:
            raise ValueError('Exactly one explicit public runtime mount required')
        result = inspect_runtime(roots.pop())
        result.update(elapsed_seconds=time.monotonic()-started,
                      declared_budget_seconds=900, source_sha256=sha(Path(__file__)))
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)
        if result['status'] != 'matched':
            raise ValueError('Public runtime checkpoint differs from official author identity')
    finally:
        timer.cancel()


if __name__ == '__main__':
    main()
