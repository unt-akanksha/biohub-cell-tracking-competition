"""Freeze the smoke head, then test all eight source movies without refitting."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'scripts/audit-trajectory-event-smoke-movies-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest()=='76711e2bb210c90c5e95641f695e4bd2baceb0c68ead57f26e2c99e512325dc2'
    model=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
    assert hashlib.sha256(model.read_bytes()).hexdigest()=='508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
    source=base.read_text(encoding='utf-8')
    changes=[
        ("name='trajectory-event-smoke-movies-v1'", "name='trajectory-event-smoke-source-eight-v1'"),
        ("stems=fit['stems']", "stems=list(read(ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features/RESULT.json')['per_movie'])"),
        ("model_sha256=sha(weights_path),fit_receipt_sha256=sha(fit_root/'RESULT.json'),",
         "model_sha256=sha(weights_path),fit_receipt_sha256=sha(fit_root/'RESULT.json'),\n                fit_movie_stems=fit['stems'],event_head_unseen_movie_stems=[s for s in stems if s not in fit['stems']],\n                event_head_refitted=False,public_backbone_independence_not_established=True,"),
        ("per_movie_summaries=helper['finite']({arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}),",
         "per_movie_summaries=helper['finite']({arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}),\n            event_head_unseen_summaries=helper['finite']({arm:scorer.summarise([r for r in values if r['stem'] not in fit['stems']]) for arm,values in rows.items()}),\n            by_embryo=helper['finite']({arm:{e:scorer.summarise([r for r in values if r['embryo']==e]) for e in ('44b6','6bba')} for arm,values in rows.items()}),")]
    for old,new in changes:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    exec(compile(source,str(base),'exec'),dict(__name__='__main__',__file__=__file__))


if __name__=='__main__':main()
