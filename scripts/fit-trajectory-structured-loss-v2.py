"""Unchanged structured objective with complete official physical source matches."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'scripts/fit-trajectory-structured-loss-v1.py'
base_sha=hashlib.sha256(base.read_bytes()).hexdigest()
assert base_sha=='9d6e7d86d6f94b29a4dd59ad08c915a9d3f7ddee2ef467a1e17b1083a1521a74'
source=base.read_text(encoding='utf-8')
changes=[
    ("('trajectory-structured-loss-v1-'+args.mode)","('trajectory-structured-loss-v2-'+args.mode)"),
    ("('trajectory-structured-loss-v1-'+args.mode+'.json')","('trajectory-structured-loss-v2-'+args.mode+'.json')"),
    (".biohub/cache/trajectory-structured-loss-v1-smoke/RESULT.json",".biohub/cache/trajectory-structured-loss-v2-smoke/RESULT.json"),
    ("    contract=dict(source_sha256=", "    physical=ROOT/'.biohub/cache/trajectory-source-label-coverage-v2'\n    physical_proof=json.loads((physical/'RESULT.json').read_text())\n    assert physical_proof['status']=='source_physical_label_coverage_audited' and physical_proof['source_only']\n    assert physical_proof['source_scope_sha256']==sha(plan) and not physical_proof['selection_or_validation_opened']\n    contract=dict(physical_label_proof_sha256=sha(physical/'RESULT.json'),base_runner_sha256=BASE_SHA,effective_runner_sha256=EFFECTIVE_SHA,source_sha256="),
    ("lp=data/(stem+'-labels.npz')","lp=physical/(stem+'-physical-labels.npz')"),
    ("sha(lp)==inv['label_sha256'][stem]","sha(lp)==physical_proof['label_sha256'][stem]"),
]
for old,new in changes:
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__,BASE_SHA=base_sha,
     EFFECTIVE_SHA=hashlib.sha256(source.encode()).hexdigest()))
