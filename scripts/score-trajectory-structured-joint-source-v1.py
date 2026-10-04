"""Same frozen official source-graph test, now for qualified structured weights."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'scripts/score-trajectory-joint-source-v1.py'
base_sha=hashlib.sha256(base.read_bytes()).hexdigest()
assert base_sha=='46e912889d62ed4cc9686586908519132957598d440e2b18c571cda2f4b34ead'
source=base.read_text(encoding='utf-8')
changes=[
    ('trajectory-joint-source-v1','trajectory-structured-joint-source-v1'),
    ('trajectory-candidate-ranker-v2-audit.json','trajectory-structured-loss-v1-full.json'),
    ("audit['qualified_for_independent_selection']","audit['qualified_for_full_movie_scoring']"),
    ("    assert sha(training/'weights.npz')==receipt['weights_sha256']",
     "    model=ROOT/'.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'\n    assert sha(model)==audit['weights_sha256']"),
    ("np.load(training/'weights.npz',allow_pickle=False)","np.load(model,allow_pickle=False)"),
    ("weights_sha256=receipt['weights_sha256']","weights_sha256=audit['weights_sha256']"),
    ('source_ranker_audit_sha256','source_structured_audit_sha256'),
    ('contract=dict(selected_model=', 'contract=dict(base_runner_sha256=BASE_SHA,effective_runner_sha256=EFFECTIVE_SHA,selected_model='),
]
for old,new in changes:
    assert old in source
    if old!='trajectory-joint-source-v1':assert source.count(old)==1
    source=source.replace(old,new)
exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__,BASE_SHA=base_sha,
     EFFECTIVE_SHA=hashlib.sha256(source.encode()).hexdigest()))
