"""Narrow, audited checkpoint substitution in the existing complete pipeline."""
import json
import os
from pathlib import Path
import subprocess

ARMS=('original','warp44','warp6')


def load_fixed_linker(model, arm, bundle, torch):
    if arm not in ARMS:raise ValueError('Unfrozen arm')
    parent=torch.load(bundle/'primary.pth',map_location='cpu',weights_only=True)
    current=parent if arm=='original' else torch.load(bundle/('44b6-final.pt' if arm=='warp44' else '6bba-final.pt'),map_location='cpu',weights_only=True)['state_dict']
    if set(current)!=set(parent):raise ValueError('Checkpoint tensor inventory changed')
    for name in parent:
        if current[name].shape!=parent[name].shape or not torch.isfinite(current[name]).all():raise ValueError('Bad checkpoint tensor')
        if not name.startswith('transformer.') and not torch.equal(current[name],parent[name]):raise ValueError('Detector/encoder changed')
    model.load_state_dict(current,strict=True);model.eval()
    pids=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
    if any(int(p)!=os.getpid() for p in pids.split()):raise RuntimeError('Foreign GPU process; stop our job without interrupting it')


def compare_control(repaired, reference):
    old=json.loads(Path(reference).read_text())
    if repaired['nodes']!=old['nodes'] or {(e['source_id'],e['target_id']) for e in repaired['edges']}!={(e['source_id'],e['target_id']) for e in old['edges']}:
        raise ValueError('Current control does not reproduce the previously validated candidate')


def build_runtime(source):
    """Preserve the public inference math and original D4 multiplicity verbatim."""
    changes=[
        ('from types import ModuleType, SimpleNamespace','from types import ModuleType, SimpleNamespace\nfrom dense_warp_movie_adapter_v1 import load_fixed_linker, compare_control'),
        ("        for arm in ('original',):", "        control_coords = None\n        for arm in ('original', 'warp44', 'warp6'):\n            load_fixed_linker(models[0], arm, args.bundle, torch)"),
        ("            filename = 'public-predictor-original.py' if arm == 'original' else 'public-predictor-d4-corrected.py'", "            filename = 'public-predictor-original.py'"),
        ("            arm_post = helper.original_postprocess(post_source) if arm == 'original' else post_source", "            arm_post = helper.original_postprocess(post_source)"),
        ("                np.savez_compressed(out / 'raw-candidates.npz', coords=coords, edges=np.asarray(edges))", "                if arm == 'original':\n                    control_coords = coords.copy()\n                else:\n                    np.testing.assert_array_equal(coords, control_coords)\n                np.savez_compressed(out / 'raw-candidates.npz', coords=coords, edges=np.asarray(edges))"),
        ("                (out / 'repair-details.json').write_text(json.dumps(repair_details, indent=2, allow_nan=False))", "                (out / 'repair-details.json').write_text(json.dumps(repair_details, indent=2, allow_nan=False))\n                if args.mode == 'full' and arm == 'original':\n                    compare_control(repaired, args.bundle / (stem + '-control.json'))"),
        ("run_id='trajectory-division-full-movie-v1'", "run_id='dense-warp-complete-movie-v1'")]
    for old,new in changes:
        if source.count(old)!=1:raise ValueError('Pinned runtime adapter drift: '+old[:65])
        source=source.replace(old,new,1)
    compile(source,'dense-warp-complete-movie-v1','exec')
    return source
