"""One-shot launch retaining fresh quota, 8h reserve and 10h watchdog checks."""
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'scripts/launch-trajectory-production-v1.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='32c33bb7ee84d009499e02a0c1111aa1ed48b80c22b1734de800218e49352671'
source=base.read_text(encoding='utf-8')
changes=[
    ('trajectory-overlap-production-v1','trajectory-structured-production-v1'),
    ('biohub-trajectory-overlap-candidate','biohub-structured-trajectory-candidate'),
    ('trajectory-overlap-kaggle-v1','trajectory-structured-kaggle-v1'),
    ('biohub-trajectory-overlap-acceptance','biohub-structured-trajectory-acceptance'),
]
for old,new in changes:
    assert old in source
    source=source.replace(old,new)
if '--overlap' not in sys.argv:sys.argv.append('--overlap')
exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__))
