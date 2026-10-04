"""Source-only ablation: separate event assignment from new coefficient training."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'scripts/audit-trajectory-event-smoke-movies-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest()=='76711e2bb210c90c5e95641f695e4bd2baceb0c68ead57f26e2c99e512325dc2'
    source=base.read_text(encoding='utf-8')
    changes=[
        ('trajectory_event_inference_v1','trajectory_event_pruned_inference_v1'),
        ("name='trajectory-event-smoke-movies-v1'","name='trajectory-event-anchor-source-v1'"),
        ("weights=arrays(weights_path)['weights'];stems=fit['stems']",
         "weights=arrays(weights_path)['anchor'];stems=list(read(ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features/RESULT.json')['per_movie'])"),
        ("model_sha256=sha(weights_path),fit_receipt_sha256=sha(fit_root/'RESULT.json'),",
         "model_sha256=sha(weights_path),parameter_key='anchor',event_head_coefficients_fitted=False,\n                ablation='frozen_old_edge_weights_and_source_prior_only',fit_receipt_sha256=sha(fit_root/'RESULT.json'),\n                fit_movie_stems=fit['stems'],event_head_unseen_movie_stems=[s for s in stems if s not in fit['stems']],blas_thread_limit=1,"),
        ("per_movie_summaries=helper['finite']({arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}),",
         "per_movie_summaries=helper['finite']({arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}),\n            event_head_unseen_summaries=helper['finite']({arm:scorer.summarise([r for r in values if r['stem'] not in fit['stems']]) for arm,values in rows.items()}),\n            by_embryo=helper['finite']({arm:{e:scorer.summarise([r for r in values if r['embryo']==e]) for e in ('44b6','6bba')} for arm,values in rows.items()}),")]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    namespace=dict(__name__='source_anchor_ablation',__file__=__file__)
    exec(compile(source,str(base),'exec'),namespace)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):namespace['main']()


if __name__=='__main__':main()
