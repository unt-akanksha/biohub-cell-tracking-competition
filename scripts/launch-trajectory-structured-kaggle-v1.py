"""Use pinned quota/reservation guard for one structured acceptance launch."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'scripts/launch-trajectory-overlap-cache-v1.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='9c59e36d036d3b8d31bf112cba373b57a37a83408607042a6a6339ecd78c20f3'
source=base.read_text(encoding='utf-8')
changes=[
    ('trajectory-overlap-cache-kaggle-v1','trajectory-structured-kaggle-v1'),
    ('biohub-trajectory-overlap-cache-acceptance-v1','biohub-structured-trajectory-acceptance-v1'),
    ('indarkarhana/biohub-trajectory-overlap-cache-acceptance','indarkarhana/biohub-structured-trajectory-acceptance'),
    ('/dev/shm/biohub-dense-warp-movie-v1-full/result.json','/dev/shm/biohub-ranker-selection-v1.LJ0JSl/full/result.json'),
]
for old,new in changes:
    assert old in source
    source=source.replace(old,new)
exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__))
