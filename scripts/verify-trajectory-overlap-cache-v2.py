"""Correct the reference platform, preserving v1's recorded A10 mismatch."""
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/verify-trajectory-kaggle-acceptance-v2.py'


def configured_source():
    source=runpy.run_path(str(ROOT/'scripts/verify-trajectory-overlap-cache-v1.py'))['configured_source']()
    start=source.index("    prior=pinned_json(ROOT/'reports/experiments/trajectory-division-full-movie-v1-result.json',")
    end=source.index('    ground_truth_opened=False\n',start)
    block='''    paired=pinned_json(ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-result.json',
        'b5a86113ee182a0dc437201f10773f72f7370115d3b622d14e4640031d08d8ec')
    if paired['status']!='acceptance_passed' or paired['contract_sha256']!='851908fa5aa8ba628aad456afa94badc6ae9e10c6f9f3e36cf43147386ef8d55':
        raise ValueError('Previously accepted paired T4 reference required')
    reference_root=ROOT/'.biohub/cache/trajectory-overlap-kaggle-v1-output'
    reference_terminal=pinned_json(reference_root/'trajectory-complete/result.json',paired['terminal_sha256'])
    if reference_terminal['torch_version']!=terminal['torch_version'] or reference_terminal['gpu_count']!=2:
        raise ValueError('Reference platform changed')
    prior=dict(paired,joint_eight_movie_rows=paired['rows'])
    identical={s:{} for s in STEMS}
    for record in paired['verified_graphs']:
        stem,arm=record['movie'],record['arm']
        if stem not in identical or arm not in ('original','repaired') or arm in identical[stem]:
            raise ValueError('Duplicated or unexpected reference graph')
        reference=pinned_json(reference_root/record['path'],record['sha256'])
        identical[stem][arm]=prepared[stem][arm]==reference
    if any(set(v)!={'original','repaired'} for v in identical.values()):
        raise ValueError('All sixteen paired T4 reference graphs required')
    exact=all(all(v.values()) for v in identical.values())
    if not exact:
        raise ValueError('Cached graph differs from paired accepted T4 graph; no rescoring')
'''
    source=source[:start]+block+source[end:]
    source=source.replace('trajectory-overlap-cache-kaggle-v1-result.json','trajectory-overlap-cache-kaggle-v2-result.json')
    source=source.replace('exact_A10_graph_identity=identical,',
        "exact_paired_T4_graph_identity=identical,reference_platform='T4',initial_A10_reference_check_failed=True,")
    compile(source,str(BASE),'exec')
    return source


if __name__=='__main__':
    exec(compile(configured_source(),str(BASE),'exec'),{'__name__':'__main__','__file__':str(BASE)})
