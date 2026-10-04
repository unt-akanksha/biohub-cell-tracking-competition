"""Verify two frozen components and a trained null-aware linker before selection."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/summarize-known-null-training.py'))
INITIAL = 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'
EXPECTED = dict(version=1,flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788',
    flow_frozen=True,displacement='child_to_parent_um_zyx',variance='unchanged_static_control',
    downsample=[1,4,4],neural_residual='preserved_then_trained')


def verify(result,terminal,split,steps,probe=None):
    identity = result['identity']
    digest = identity.get('frozen_flow_sha256','')
    if (identity.get('image_motion') != EXPECTED or result.get('flow_unchanged') is not True
        or len(digest)!=64 or set(digest)-set('0123456789abcdef')
        or terminal.get('run_id') != 'image-motion-linker-v1' or terminal.get('declared_budget_seconds') != 3600):
        raise ValueError('Exact frozen image-flow identity and runtime receipts required')
    report = BASE['verify'](result,terminal,split,steps,probe,initial=INITIAL)
    if probe is not None and identity['frozen_flow_sha256'] != probe['identity']['frozen_flow_sha256']:
        raise ValueError('Frozen image-flow weights changed between probe and fit')
    report.update(status='verified_image_motion_linker_training_not_selection',flow_unchanged=True,
                  frozen_flow_sha256=digest,image_motion=EXPECTED)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=int,choices=(1,2),required=True)
    args = parser.parse_args()
    cache = ROOT/'.biohub/cache/kernel-outputs'/('image-motion-linker-probe-v1' if args.version == 1 else 'image-motion-linker-fit-v2')/'image_motion_linker'
    paths = dict(result=cache/'outputs/result.json',terminal=cache/'launcher_terminal.json')
    probe_path = ROOT/'.biohub/cache/kernel-outputs/image-motion-linker-probe-v1/image_motion_linker/outputs/result.json'
    report = verify(*(json.loads(paths[k].read_text()) for k in ('result','terminal')),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()),
        100 if args.version == 1 else 1000,None if args.version == 1 else json.loads(probe_path.read_text()))
    report['source_sha256'] = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/f'reports/experiments/image-motion-linker-v{args.version}-training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','source_sha256')},indent=2))
