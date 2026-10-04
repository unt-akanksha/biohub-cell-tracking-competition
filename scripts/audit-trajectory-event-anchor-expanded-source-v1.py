"""Evaluate the unchanged anchor on all 52 additional source movies.

Reuses the pinned full-movie scorer and artifact verification, not its fitted
pilot model. No gate is fitted, no validation labels are opened, and source
failures remain visible. The already evaluated batch zero is not rerun.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_source():
    base = ROOT / 'scripts/audit-trajectory-event-pilot-source-v2.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == 'b2a83ceaf484d7388062663a9ed4e99e3df21082e7e535f2124586aa33496f5d'
    source = base.read_text(encoding='utf-8')
    start = source.index("    fit_root=ROOT/")
    stop = source.index("    runtime=ROOT/", start)
    source = source[:start] + """    fit_root=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1'
    fitted=read(fit_root/'RESULT.json')
    model=fit_root/'epoch-3.npz'
    assert sha(model)=='508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
    weights=arrays(model)['anchor']
    assert np.isfinite(weights).all() and weights.shape==(30,)
    # The anchor uses old source edge weights and a prior from two batch-0
    # movies. It does not use the fitted smoke/pilot event coefficients.
    old_source=read(ROOT/'reports/experiments/trajectory-event-anchor-source-v1.json')
    assert old_source['parameter_key']=='anchor' and not old_source['event_head_coefficients_fitted']
    fit_stems=set(old_source['fit_movie_stems'])
""" + source[stop:]
    replacements = [
        ("name='trajectory-event-pilot-source-v2'", "name='trajectory-event-anchor-expanded-source-v1'"),
        ("    fit_stems=set(contract['source_stems']);movies=[];inputs={}", "    movies=[];inputs={}"),
        ("    for batch in (0,1):", "    for batch in range(1,8):"),
        ("    assert len(movies)==16 and len(fit_stems)==7 and fit_stems<=set(movies)",
         "    assert len(movies)==52 and len(fit_stems)==2 and not fit_stems.intersection(movies)"),
        ("fit_contract_sha256=sha(fit_root/'CONTRACT.json'),fit_result_sha256=sha(fit_root/'RESULT.json'),",
         "parameter_key='anchor',event_head_coefficients_fitted=False,movie_id_routing=False,\n        prior_source_report_sha256=sha(ROOT/'reports/experiments/trajectory-event-anchor-source-v1.json'),"),
        ("event_pilot", "anchor"),
        ("full_source_pilot_diagnostic_complete", "expanded_source_anchor_diagnostic_complete"),
    ]
    for old,new in replacements:
        assert old in source,old
        source=source.replace(old,new)
    # All 52 movies are head-prior-held-out: no empty fitted-cohort summary.
    old="fit_summaries=helper['finite'](summaries(lambda r:not r['head_held_out'])),"
    assert source.count(old)==1
    source=source.replace(old,"no_event_prior_fit_movies_in_this_cohort=True,")
    # Pin the official scorer's implementation as well as GT artifact hashes.
    marker="        helper=runpy.run_path("
    assert source.count(marker)==1
    source=source.replace(marker,"        report['all_predictions_frozen_before_scoring']=True;persist()\n"+marker)
    return source


if __name__=='__main__':
    source=build_source()
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
