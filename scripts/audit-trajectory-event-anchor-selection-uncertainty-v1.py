"""Paired whole-movie uncertainty diagnostic; no tuning or gate replacement."""
import json
from pathlib import Path
import runpy
import sys
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha


def main():
    path=ROOT/'reports/experiments/trajectory-event-anchor-selection-v1.json'
    assert sha(path)=='43e94177fe98c2ade5925aa78fd5c892a420a7d8323d1dfa986c75885d240fcb'
    report=json.loads(path.read_text(encoding='utf-8'))
    assert report['status']=='fixed_anchor_selection_diagnostic_complete'
    output=ROOT/'reports/experiments/trajectory-event-anchor-selection-uncertainty-v1.json'
    assert not output.exists()
    scorer=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))['load_scorer']()
    rows=report['rows'];baseline=rows['baseline'];anchor=rows['anchor']
    assert [r['stem'] for r in baseline]==[r['stem'] for r in anchor]
    assert len(baseline)==10 and all(np.isfinite(r['adj_edge_jaccard']) for a in rows.values() for r in a)
    rng=np.random.default_rng(20260914);draws=5000
    strata={e:np.array([i for i,r in enumerate(baseline) if r['embryo']==e]) for e in ('44b6','6bba')}
    def delta(indices):
        return scorer.summarise([anchor[int(i)] for i in indices])['score']-scorer.summarise([baseline[int(i)] for i in indices])['score']
    assert np.isclose(delta(range(10)),report['summaries']['anchor']['score']-report['summaries']['baseline']['score'],rtol=0,atol=1e-12)
    distributions={k:[] for k in ('pooled_fixed_embryo_mix','44b6','6bba')}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='No divisions present across any sample')
        for _ in range(draws):
            sampled={e:rng.choice(indices,size=len(indices),replace=True) for e,indices in strata.items()}
            distributions['pooled_fixed_embryo_mix'].append(delta(np.concatenate(list(sampled.values()))))
            for e,indices in sampled.items():distributions[e].append(delta(indices))
    summary={k:dict(percentile_interval_95=np.quantile(v,[.025,.975]).tolist(),
                    fraction_of_bootstrap_draws_positive=float(np.mean(np.array(v)>0))) for k,v in distributions.items()}
    worst=[]
    for i in range(10):
        worst.append(dict(stem=baseline[i]['stem'],delta=delta([i]),
            delta_tp=anchor[i]['edge_tp']-baseline[i]['edge_tp'],
            delta_fp=anchor[i]['edge_fp']-baseline[i]['edge_fp'],
            delta_fn=anchor[i]['edge_fn']-baseline[i]['edge_fn']))
    result=dict(status='selection_movie_uncertainty_diagnostic_complete',input_sha256=sha(path),
        source_sha256=sha(Path(__file__)),draws=draws,seed=20260914,
        sampling_unit='paired complete movie, stratified by the two observed embryos',
        conditional_on_observed_embryos_not_new_embryo_generalization=True,
        prior_pipeline_selection_exposure_disclosed=True,intervals=summary,
        worst_movies=sorted(worst,key=lambda r:r['delta']),selection_failure_preserved=True,
        original_quality_checks=report['quality_checks'],no_model_or_threshold_selected=True,
        bootstrap_fraction_is_not_probability_of_leaderboard_improvement=True,
        authorized_for_submission=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],intervals=summary)),flush=True)


if __name__=='__main__':main()
