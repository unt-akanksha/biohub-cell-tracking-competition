"""Stage one small cached-centroid/owned-flow gate; never overwrite staging."""
import ast
import json
from pathlib import Path
import runpy
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN = 'focus-owned-flow-probe-v1'
TARGET = ROOT/'kaggle'/('biohub-'+RUN)


def build():
    nb, meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-selection.py'))['build']()
    source = ''.join(nb['cells'][1]['source'])
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(node.value)
    runtime = {k:v for k,v in runtime.items() if not k.startswith('tests/')}
    runtime['run_pilot.py'] = (ROOT/'scripts/run-focus-owned-flow-probe.py').read_text()
    for name in ('raw_centroid_flow_sampling','independent_motion_prior','backward_flow_linking'):
        runtime[name+'.py'] = (ROOT/f'research/{name}.py').read_text().replace(
            'from research.independent_motion_prior import','from independent_motion_prior import')
    runtime['tests/test_raw_centroid_flow_sampling.py'] = (
        ROOT/'tests/test_raw_centroid_flow_sampling.py').read_text().replace(
            'from research.raw_centroid_flow_sampling import','from raw_centroid_flow_sampling import')
    nb['cells'][1]['source'] = source.replace(ast.get_source_segment(source,node),
        'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('backward-flow-selection-v1',RUN)
        source = source.replace('backward_flow_selection','focus_owned_flow_probe')
        if index == len(nb['cells'])-1:
            old = "reference = mounted('biohub-independent-joint-selection-v1','independent_joint_selection')"
            if source.count(old) != 1:
                raise ValueError('Raw input mount injection anchor changed')
            source = source.replace(old,"reference = mounted('biohub-focus3d-raw-detections-v1','')")
            source = source.replace('import signal',
                "subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests'),'-q'],check=True,cwd=str(runtime))\nimport signal")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id=RUN,frames=3,training_stem='6bba_23af9eeb',
        raw_terminal_sha256='0609934b1e2a40473763acf521cfcf7120e418f5857c24d6f28c0e662638bd41',
        detector_inference=False,public_linker_loaded=False,ground_truth_opened=False,
        sampling='exact centroids; declared trailing-border constant flow extension')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Focus Owned Flow Probe v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=[
            'indarkarhana/biohub-backward-flow-fit-v1/1',
            'indarkarhana/biohub-focus3d-raw-detections-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        nb,meta=build()
        print(json.dumps(dict(run_id=RUN,gpu=meta['enable_gpu'],target=str(TARGET))))
        raise SystemExit(0)
    if TARGET.exists():
        raise ValueError('Refuse to overwrite frozen cached-centroid probe')
    nb,meta = build(); TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
