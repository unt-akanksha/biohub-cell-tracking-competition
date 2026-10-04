"""Compose source-only flow inference from the frozen successful four-movie run."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUN = 'focus-source-flow-v1'
SLUG = 'biohub-' + RUN


def build(cache_payload):
    policy = runpy.run_path(str(ROOT / 'research/focus_source_flow_contract.py'))['receipt'](
        cache_payload, (ROOT / 'research/independent_real_baseline_v1_split.json').read_bytes())
    parent = ROOT / 'kaggle/biohub-focus-owned-flow-full-v1/biohub-focus-owned-flow-full-v1.ipynb'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != '5af97d02de71448af3d4f2367db307a77212fb59ae0d1389f0a6beffb7e3db9f':
        raise ValueError('Frozen completed native-flow implementation required')
    nb = json.loads(parent.read_text())
    source = ''.join(nb['cells'][1]['source'])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(node.value)
    runtime['run_pilot.py'] = (ROOT / 'scripts/run-focus-source-flow.py').read_text()
    runtime['focus_source_flow_contract.py'] = (ROOT / 'research/focus_source_flow_contract.py').read_text()
    runtime['verified_source_cache.json'] = cache_payload.decode()
    nb['cells'][1]['source'] = source.replace(ast.get_source_segment(source, node), 'runtime_sources = ' + repr(runtime)).splitlines(keepends=True)
    for index in (0, len(nb['cells']) - 1):
        text = ''.join(nb['cells'][index]['source']).replace('focus-owned-flow-full-v1', RUN)
        text = text.replace('focus_owned_flow_full', 'focus_source_flow')
        if index == len(nb['cells']) - 1:
            before = "reference = mounted('biohub-focus3d-raw-detections-v1','')"
            if text.count(before) != 1:
                raise ValueError('Raw source mount anchor changed')
            text = text.replace(before, "reference = mounted('biohub-focus-source-cache-v1','')")
        nb['cells'][index]['source'] = text.splitlines(keepends=True)
    nb['metadata']['codex'] = dict(run_id=RUN, contract=policy, declared_budget_seconds=3600,
                                  authorized_for_submission=False, ground_truth_opened=False)
    meta = json.loads((parent.parent / 'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/' + SLUG, title=SLUG, code_file=SLUG + '.ipynb',
                kernel_sources=['indarkarhana/biohub-backward-flow-fit-v1/1',
                                'indarkarhana/biohub-focus-source-cache-v1/1',
                                'indarkarhana/biohub-focus-owned-flow-probe-v1/1'])
    for c in nb['cells']:
        ast.parse(''.join(c['source']))
    return nb, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'reports/experiments/focus-source-cache-v1-result.json'
    payload = path.read_bytes()
    actual = runpy.run_path(str(ROOT / 'scripts/verify-focus-source-cache.py'))['verify'](
        ROOT / '.biohub/cache/kernel-outputs/focus-source-cache-v1')
    if actual != json.loads(payload):
        raise ValueError('Actual source cache differs from verified report')
    nb, meta = build(payload)
    if args.check:
        print(json.dumps(dict(run_id=RUN, gpu=meta['enable_gpu'])))
        raise SystemExit(0)
    target = ROOT / 'kaggle' / SLUG
    if target.exists():
        raise ValueError('Refuse to overwrite frozen source-flow notebook')
    target.mkdir()
    (target / meta['code_file']).write_text(json.dumps(nb))
    (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(target)
