"""Package only exact model/runtime and four already-verified replay packets."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUN='antelume-focus-head-replay-v1'


def digest(data):return hashlib.sha256(data).hexdigest()


def build():
    target=ROOT/'.biohub/cache'/RUN
    if target.exists():raise ValueError('Never overwrite staged cloud probe')
    nbpath=ROOT/'kaggle/biohub-focus-adaptation-features-v1/biohub-focus-adaptation-features-v1.ipynb'
    if digest(nbpath.read_bytes())!='502bcc24048eb779210f00e3d4874087dd3f94a9c2c02fa966ca9cbbef9bc8df':raise ValueError('Exact original feature notebook required')
    nb=json.loads(nbpath.read_text());nodes=ast.parse(''.join(nb['cells'][1]['source'])).body
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('sources','runtime_sources')}
    files={'repo/'+p:v.encode() for p,v in assignments['sources'].items()}
    files['runtime/independent_real_baseline.py']=assignments['runtime_sources']['independent_real_baseline.py'].encode()
    files['run_probe.py']=(ROOT/'scripts/probe-antelume-focus-head.py').read_bytes()
    checkpoint=ROOT/'.biohub/cache/kernel-outputs/image-motion-linker-cloud-transfer-v1/image_motion_linker/outputs/last.pt'
    if digest(checkpoint.read_bytes())!='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144':raise ValueError('Exact original owned checkpoint required')
    files['checkpoint.pt']=checkpoint.read_bytes();pairs=[]
    cache=ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs'
    receipt=json.loads((ROOT/'reports/experiments/focus-adaptation-features-v1-result.json').read_text())
    for stem in ('6bba_f1fde7e0','6bba_23af9eeb'):
        record=next(r for r in receipt['records'] if r['stem']==stem)
        manifest_path=cache/stem/'manifest.json'
        if digest(manifest_path.read_bytes())!=record['manifest_sha256']:raise ValueError('Original replay manifest changed')
        manifest=json.loads(manifest_path.read_text())
        reference=ROOT/'.biohub/cache/kernel-outputs/focus-owned-neural-probe-v1/focus_owned_neural_probe/outputs'/(stem+'.npz')
        item=next(r for r in json.loads(assignments['runtime_sources']['features_spec.json'])['movies'] if r['stem']==stem)
        if digest(reference.read_bytes())!=item['probe_sha256']:raise ValueError('Exact Kaggle matrix reference required')
        refname='reference/'+stem+'.npz';files[refname]=reference.read_bytes()
        for t,row in enumerate(manifest['pairs']):
            packet=cache/stem/row['file'];name='packets/'+stem+'/'+row['file']
            if digest(packet.read_bytes())!=row['sha256']:raise ValueError('Original cached packet changed')
            files[name]=packet.read_bytes();pairs.append(dict(stem=stem,frame=t,packet=name,reference=refname))
    if len(pairs)!=4 or sum(map(len,files.values()))>40_000_000:raise ValueError('Small four-packet transfer only')
    manifest=dict(run_id=RUN,files={p:digest(v) for p,v in files.items()},pairs=pairs,
        model_tensor_sha256='41e82ccfd2e049abbc3d7d11e5eb08b60361b3673b606f848787597de6d62327',
        declared_wall_seconds=180,source_selection_opened=False,new_target_movies_opened=0)
    files['manifest.json']=json.dumps(manifest,indent=2).encode();target.mkdir()
    archive=target/'probe.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as out:
        for p,v in files.items():out.writestr(p,v)
    identity=dict(run_id=RUN,archive_sha256=digest(archive.read_bytes()),bytes=archive.stat().st_size,
        builder_sha256=digest(Path(__file__).read_bytes()),worker_sha256=digest(files['run_probe.py']),files=len(files))
    (target/'staged_identity.json').write_text(json.dumps(identity,indent=2));print(json.dumps(identity))


if __name__=='__main__':build()
