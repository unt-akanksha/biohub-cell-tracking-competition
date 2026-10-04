"""Observe only the already-launched final version2 and harvest its evidence."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'scripts/watch-trajectory-event-anchor-production-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == '356ddd8d76f39cd9eb04fe77788c2fbcaa7f5c1af7a03f7ee2d7b5b0c8ac675b'
    source = base.read_text(encoding='utf-8')
    source = source.replace('trajectory-event-anchor-production-v1', 'trajectory-event-anchor-final-v2')
    changes = [
        ("and launch['public_test_only']", "and launch['platform_timeout_seconds']==43200"),
        ('version=1,observer_pid', 'version=2,observer_pid'),
        ("actual['current_version_number']==1", "actual['current_version_number']==2"),
        ('deadline=time.monotonic()+5400', 'deadline=time.monotonic()+45000'),
    ]
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=__file__))


if __name__ == '__main__':
    main()
