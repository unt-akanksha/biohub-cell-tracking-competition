"""Specialize pinned production staging for the accepted structured model."""
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'scripts/prepare-trajectory-production-v1.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='0ebb2a17b36c68bf4280b37af773f7df12106d822721ce56a85aa9d4d92ba9e1'
source=base.read_text(encoding='utf-8')
changes=[
    ('trajectory-overlap-kaggle-v1','trajectory-structured-kaggle-v1'),
    ('biohub-trajectory-overlap-candidate','biohub-structured-trajectory-candidate'),
    ('biohub-trajectory-overlap-acceptance-v1','biohub-structured-trajectory-acceptance-v1'),
    ('trajectory-overlap-acceptance.ipynb','structured-trajectory-acceptance.ipynb'),
    ('trajectory-overlap-production-v1','trajectory-structured-production-v1'),
    ('trajectory-motion-candidate.ipynb','structured-trajectory-candidate.ipynb'),
    ('Biohub Trajectory Motion Candidate','Biohub Structured Trajectory Candidate'),
    ('Biohub source-trained trajectory motion candidate. Public LF-DCTTA backbone plus our fixed motion-expert endpoint repair. Offline two-GPU whole-movie inference; all input movies required. See runtime NOTICE and PROVENANCE.',
     'Biohub structured trajectory candidate. Licensed image-model ensemble plus our source-trained endpoint repair and frozen structured assignment model. Offline two-T4 whole-movie inference; all input movies required. See runtime NOTICE and PROVENANCE.'),
]
for old,new in changes:
    assert old in source
    source=source.replace(old,new)
if '--overlap' not in sys.argv:sys.argv.append('--overlap')
exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__))
