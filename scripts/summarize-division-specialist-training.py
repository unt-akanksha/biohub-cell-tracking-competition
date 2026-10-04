"""Verify both real rare-event exposure and frozen-detector training receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/summarize-known-null-training.py'))
INITIAL = 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'


def verify(result,terminal,split,steps,probe=None):
    identity = result['identity']
    sampling = identity.get('division_sampling',{})
    if (identity.get('division_specialist') != dict(version=1,division_window_mass=.5)
        or sampling.get('division_windows') != 114 or sampling.get('ordinary_windows') != 11509
        or not math.isclose(sampling.get('division_mass',0),.5,rel_tol=0,abs_tol=1e-12)
        or sampling.get('replacement') is not True
        or sampling.get('weights_sha256') != '05390c6f1050f5d6e4a269f9e975ebaf104806d49f68a8d8afc854535b85c106'):
        raise ValueError('Exact frozen training-only division sampler required')
    report = BASE['verify'](result,terminal,split,steps,probe,initial=INITIAL,
                           loss='division_balanced_with_annotated_missing_parent_null_v1')
    total = sum(r['division_columns'] for r in result['history'])
    if total <= 0 or result['division_supervised_total'] != total:
        raise ValueError('Missing real matched daughter supervision')
    report.update(division_columns=total,division_sampling=sampling,
        correct_division_columns=sum(r['correct_division_columns'] for r in result['history']),
        profile='Frozen detector; case-balanced divisions and retained annotated nulls')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=int,choices=(1,2),required=True)
    args = parser.parse_args()
    cache = ROOT/'.biohub/cache/kernel-outputs'/('division-specialist-probe-v1' if args.version == 1 else 'division-specialist-fit-v2')/'independent_division_specialist'
    result_path = cache/'outputs/result.json'
    terminal_path = cache/'launcher_terminal.json'
    probe_path = ROOT/'.biohub/cache/kernel-outputs/division-specialist-probe-v1/independent_division_specialist/outputs/result.json'
    report = verify(json.loads(result_path.read_text()),json.loads(terminal_path.read_text()),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()),
        100 if args.version == 1 else 1000,None if args.version == 1 else json.loads(probe_path.read_text()))
    report['source_sha256'] = {key:hashlib.sha256(path.read_bytes()).hexdigest()
                              for key,path in [('result',result_path),('terminal',terminal_path)]}
    (ROOT/f'reports/experiments/division-specialist-v{args.version}-training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','source_sha256')},indent=2))
