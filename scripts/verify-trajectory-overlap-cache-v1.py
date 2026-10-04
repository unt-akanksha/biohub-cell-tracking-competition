"""Reuse the pinned eight-movie verifier, requiring exact cached graph identity."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scripts/verify-trajectory-kaggle-acceptance-v2.py'
BASE_SHA = '57cdb095440dad855428149feeba094f62372662dcb62f6ec2e68eca9f84f7e2'


def configured_source():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    source = BASE.read_text(encoding='utf-8')
    replacements = (
        ('args=p.parse_args();start=time.monotonic()',
         'args=p.parse_args();args.overlap=True;start=time.monotonic()'),
        ("CONTRACT='851908fa5aa8ba628aad456afa94badc6ae9e10c6f9f3e36cf43147386ef8d55'",
         "CONTRACT='99c9fb48404e31051ce3096a0901b32698d19742abb3f04d5f4dc0453f9ec411'"),
        ('trajectory-overlap-kaggle-v1-result.json', 'trajectory-overlap-cache-kaggle-v1-result.json'),
        ('    ground_truth_opened=False\n',
         "    if not exact:\n        raise ValueError('Exact-cache runtime changed a graph; reject without rescoring')\n    ground_truth_opened=False\n"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Pinned verifier structure changed')
        source = source.replace(old, new)
    compile(source, str(BASE), 'exec')
    return source


if __name__ == '__main__':
    exec(compile(configured_source(), str(BASE), 'exec'), {'__name__': '__main__', '__file__': str(BASE)})
