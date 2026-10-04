"""Stage a frozen offline CPU identity check, not inference or a submission."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'biohub-focus-runtime-identity-v1'


def build():
    return dict(id='indarkarhana/' + SLUG, title=SLUG,
                code_file='audit-focus-runtime-identity.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False,
                enable_tpu=False, enable_internet=False,
                dataset_sources=['qiweiyin/focus3d-nuclei-runtime'],
                competition_sources=[], kernel_sources=[], model_sources=[])


if __name__ == '__main__':
    folder = ROOT / 'kaggle' / SLUG
    if folder.exists():
        raise ValueError('Refuse to overwrite a staged runtime identity audit')
    folder.mkdir()
    (folder / 'audit-focus-runtime-identity.py').write_bytes((ROOT / 'scripts/audit-focus-runtime-identity.py').read_bytes())
    (folder / 'kernel-metadata.json').write_text(json.dumps(build(), indent=2))
    print(folder)
