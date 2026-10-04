"""Stage a private offline CPU-only metadata check."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'biohub-focus-checkpoint-metadata-v1'


if __name__ == '__main__':
    target = ROOT / 'kaggle' / SLUG
    if target.exists():
        raise ValueError('Refuse to overwrite checkpoint metadata audit')
    meta = json.loads((ROOT / 'kaggle/biohub-focus-runtime-identity-v1/kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/' + SLUG, title=SLUG, code_file='audit-focus-checkpoint-metadata.py')
    if any(meta[k] for k in ('enable_gpu', 'enable_tpu', 'enable_internet', 'competition_sources', 'kernel_sources')):
        raise ValueError('Metadata check must remain offline CPU-only, without competition data')
    target.mkdir()
    (target / meta['code_file']).write_bytes((ROOT / 'scripts/audit-focus-checkpoint-metadata.py').read_bytes())
    (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(target)
