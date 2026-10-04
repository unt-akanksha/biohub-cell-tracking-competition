"""Specialize pinned terminal-only, exact-file-set artifact recovery."""
import hashlib
from pathlib import Path

base=Path(__file__).with_name('harvest-trajectory-disagreement-source-v1.py')
assert hashlib.sha256(base.read_bytes()).hexdigest()=='df056bca671d405b0e5c0c814632217b9254c7d53ee3b32290ab7cab444f822f'
source=base.read_text(encoding='utf-8')
source=source.replace('/dev/shm/biohub-disagreement-source-v1.nRsU8k/',
                      '/dev/shm/biohub-ranker-selection-v1.LJ0JSl/')
source=source.replace('trajectory-disagreement-source-v1','trajectory-ranker-selection-v1')
exec(compile(source,str(base),'exec'),{'__name__':'__main__','__file__':str(base)})
