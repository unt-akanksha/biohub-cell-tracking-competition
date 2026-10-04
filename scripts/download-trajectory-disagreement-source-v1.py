"""Pinned eight-movie specialization of the smoke-tested ZIP range downloader."""
import hashlib
from pathlib import Path

STEMS=('44b6_cf8fed6b','44b6_f28707c6','44b6_c50204e0','44b6_cf2536e8',
       '6bba_78a7bd97','6bba_5dfe9ad1','6bba_acd782a8','6bba_969618f6')


def configured_source(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='65bc09ef9734a41173623fd46bfa0731eaa86e1cde5b6336607be0cb3313e09c'
    source=path.read_text(encoding='utf-8')
    replacements=[
        ("STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')",'STEMS = '+repr(STEMS)),
        ('    plan = json.loads(payload)',
         "    plan = json.loads(payload)\n    if plan['movie_plan_sha256']!='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf':\n        raise ValueError('Source-only movie plan changed')\n    plan.update(stems=list(STEMS),ground_truth_included=False,total_bytes=plan['total_image_bytes'])"),
        ('len(records) != 408','len(records) != 816'),
        ('total=408','total=816'),
        ("plan['total_bytes'] > 3 * 1024 ** 3","plan['total_bytes'] > 5 * 1024 ** 3"),
        ("run_id='trajectory-division-archive-v1'","run_id='trajectory-disagreement-source-v1'"),
    ]
    for old,new in replacements:
        assert source.count(old)==1
        source=source.replace(old,new)
    compile(source,str(path),'exec')
    return source


if __name__=='__main__':
    base=Path(__file__).with_name('download-trajectory-division-archive-v1.py')
    exec(compile(configured_source(base),str(base),'exec'),{'__name__':'__main__','__file__':str(base)})
