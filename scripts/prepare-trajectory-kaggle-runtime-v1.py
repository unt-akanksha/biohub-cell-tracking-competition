"""Freeze reviewed model/code/wheel bundle and private two-T4 acceptance notebook."""
from __future__ import annotations
import ast
import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_portable_adapter_v1 import portable_source
from research.trajectory_runtime_v1 import sha, verify_bundle
from research.submission_sharding import validate_submission_kernel_metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision',type=int,default=2)
    revision=parser.parse_args().revision
    if revision < 2: raise ValueError('Revision1 retained as incomplete dependency closure')
    source=ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle'
    if sha(source/'CONTRACT.json') != '7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1':
        raise ValueError('Validated base contract changed')
    prior=json.loads((source/'CONTRACT.json').read_text())
    quality=ROOT/'reports/experiments/trajectory-division-full-movie-v1-result.json'
    if sha(quality) != '6dc733376a89e0eb8fdb8b996caf008252c84799199d35ef5f7f48a6ff39bb40':
        raise ValueError('Quality evidence changed')
    if json.loads(quality.read_text())['status'] != 'diagnostic_pass':
        raise ValueError('Eight-movie gate required')
    bundle=ROOT/f'.biohub/cache/trajectory-kaggle-runtime-v{revision}-bundle'
    upload=ROOT/f'.biohub/staging/biohub-trajectory-motion-runtime-upload-v{revision}'
    notebook=ROOT/f'.biohub/staging/biohub-trajectory-motion-acceptance-v{revision}'
    for folder in (bundle,upload,notebook):
        if folder.exists(): raise ValueError(f'Preserve frozen staging: {folder.name}')
    bundle.mkdir(); upload.mkdir(); notebook.mkdir()
    for name,digest in prior['bundle_sha256'].items():
        if sha(source/name)!=digest: raise ValueError('Validated source changed')
        if name in ('public-predictor-d4-corrected.py','run-trajectory-division-full-movie-v1.py'):
            continue
        shutil.copyfile(source/name,bundle/name)
        if name.startswith(('public-predictor-', 'public-postprocess-')):
            text=(bundle/name).read_text()
            notice=('# Modified/extracted for Biohub trajectory runtime, 2026-09-13.\n'
                    '# Source: sjlee101/biohub-lf-dctta v1, Apache-2.0.\n'
                    '# Selected definitions only; log-path adapter and ORIGINAL D4 reversal\n'
                    '# are applied by the worker. See NOTICE.txt and licenses/Apache-2.0.txt.\n')
            updated=notice+text
            if ast.dump(ast.parse(text)) != ast.dump(ast.parse(updated)):
                raise ValueError('Attribution notice changed public code semantics')
            (bundle/name).write_text(updated)
    original=(source/'run-trajectory-division-full-movie-v1.py').read_text()
    (bundle/'portable-worker.py').write_text(portable_source(original))
    for relative in ('research/trajectory_runtime_v1.py','research/submission_sharding.py',
                     'scripts/run-trajectory-two-gpu-v1.py'):
        shutil.copyfile(ROOT/relative,bundle/Path(relative).name)
    wheel_root=ROOT/'.biohub/cache/trajectory-runtime-wheels-v1'
    wheels=sorted(wheel_root.glob('*.whl'))
    if len(wheels)!=64: raise ValueError('Expected exact64 wheel closure including HTTP2 extras')
    (bundle/'wheels').mkdir(); (bundle/'licenses').mkdir()
    wheel_records=[]
    for path in wheels:
        shutil.copyfile(path,bundle/'wheels'/path.name)
        with zipfile.ZipFile(path) as archive:
            metadata_names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata_names)!=1: raise ValueError('Invalid wheel metadata')
            metadata=BytesParser().parsebytes(archive.read(metadata_names[0]))
            license_files=[n for n in archive.namelist() if
                any(token in Path(n).name.lower() for token in ('license','copying','notice','copyright'))]
            wheel_records.append(dict(filename=path.name,sha256=sha(path),bytes=path.stat().st_size,
                distribution=metadata['Name'],version=metadata['Version'],
                license_expression=metadata.get('License-Expression'),
                license_field=metadata.get('License'),license_files=license_files))
            if path.name.startswith('google_crc32c-'):
                name=next(n for n in license_files if n.endswith('/LICENSE'))
                (bundle/'licenses/Apache-2.0.txt').write_bytes(archive.read(name))
    shutil.copyfile(ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/LICENSE',bundle/'licenses/official-BSD-3-Clause.txt')
    licenses={}
    for kind in ('primary','secondary','deepcenter'):
        path=ROOT/f'.biohub/cache/trajectory-license-audit-v1/{kind}/dataset-metadata.json'
        data=json.loads(path.read_text())['info']
        if data['licenses'] != [dict(name='CC0-1.0')]: raise ValueError('Public weights license changed')
        licenses[kind]=dict(url=f"https://www.kaggle.com/datasets/{data['ownerUser']}/{data['datasetSlug']}",
                            license='CC0-1.0',metadata_sha256=sha(path))
        shutil.copyfile(path,bundle/'licenses'/f'{kind}-dataset-metadata.json')
    notice='''Biohub trajectory motion candidate v1 (2026-09-13)

Public LF-DCTTA model/predictor/postprocess definitions: sjlee101,
https://www.kaggle.com/code/sjlee101/biohub-lf-dctta , pinned v1 notebook SHA
95f08bb82388e9206f45c86daf926bb3d7a7fff9889c89b5fa9596f8e13c3bd4.
Public Kaggle notebooks: Apache-2.0; full license included.
Reviewed AST definitions only are executed; no notebook proxy/GT/search cells.
Modified files public-predictor-original.py and public-postprocess-d4-corrected.py
are extracted definitions. The latter stores the historical D4 diagnostic
variant but original_postprocess() reverses it to the exact ORIGINAL baseline.
Predictor changes only relocate logs. The rejected D4 fix is NOT used.

Public checkpoint/model-code datasets: pilkwang; exact CC0-1.0 metadata included.
Official model-source ancestry: royerlab/kaggle-cell-tracking-competition,
Copyright (c) 2026, Thibaut Goldsborough; BSD-3-Clause notice retained.
No public prediction tables, ground truth or scorer are in this runtime.

Our addition: two source-trained robust AR(2) motion-support experts and a fixed
reciprocal genuine-endpoint union. Public detectors/linkers are not claimed as
our original work. New runtime adapters pin SCIP (the validated backend), isolate
two CUDA workers, verify complete movies and assemble checked CSV output.
New project-owned code is distributed under Apache-2.0 for this runtime.
Dependency wheels are unmodified and retain their own licenses/notices.
Torch/CUDA come from the pinned Kaggle base image, NOT from redistributed wheels.
Whole-model validation is not pristine: public-backbone overlap and historically
exposed movies remain limitations. The0.9453907 local diagnostic is not an LB score.
'''
    (bundle/'NOTICE.txt').write_text(notice)
    (bundle/'WHEEL_MANIFEST.json').write_text(json.dumps(wheel_records,indent=2))
    (bundle/'PROVENANCE.json').write_text(json.dumps(dict(public_datasets=licenses,
        prior_contract_sha256=sha(source/'CONTRACT.json'),
        prior_worker_sha256=sha(source/'run-trajectory-division-full-movie-v1.py'),
        adapter_sha256=sha(ROOT/'research/trajectory_portable_adapter_v1.py'),
        quality_sha256=sha(quality),local_score=.9453907265031998,
        public_leaderboard_score=None,public_backbone_training_overlap=True,
        motion_validation_movies_excluded=True),indent=2))
    for path in bundle.glob('*.py'): ast.parse(path.read_text())
    pins={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    contract=dict(run_id='trajectory-kaggle-runtime-v1',packaging_revision=revision,bundle_sha256=pins,
                  ground_truth_included=False,public_prediction_tables_included=False,
                  source_mixture_unchanged=True,quality_receipt_sha256=sha(quality),
                  required_gpus=2,internet_enabled=False,maximum_inference_seconds=36000,
                  production_runtime_acceptance_passed=False,authorized_for_submission=False)
    (bundle/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
    contract_sha=sha(bundle/'CONTRACT.json'); verify_bundle(bundle,contract_sha)
    archive_path=upload/'trajectory-runtime-v1.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.write(path,path.relative_to(bundle).as_posix())
    archive_sha=sha(archive_path)
    (upload/'dataset-metadata.json').write_text(json.dumps(dict(
        title='Biohub Trajectory Motion Runtime v1',id='indarkarhana/biohub-trajectory-motion-runtime-v1',
        licenses=[dict(name='apache-2.0')],isPrivate=True,
        description='Private offline reproducible runtime. Upstream weights CC0, notebook Apache2, model ancestry BSD3, unmodified wheels retain individual licenses. See NOTICE and PROVENANCE.'),indent=2))
    bootstrap=(ROOT/'scripts/trajectory-kaggle-bootstrap-v1.py').read_text()
    bootstrap=bootstrap.replace('__ARCHIVE_SHA256__',archive_sha).replace('__CONTRACT_SHA256__',contract_sha).replace('__RUN_MODE__','acceptance')
    ast.parse(bootstrap)
    notebook_json=dict(cells=[dict(cell_type='markdown',metadata={},source=['Offline two-T4 runtime acceptance. No submission.csv and no truth access.']),
                              dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=bootstrap.splitlines(keepends=True))],
                       metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                     language_info=dict(name='python',version='3.12.13')),nbformat=4,nbformat_minor=5)
    (notebook/'trajectory-motion-acceptance.ipynb').write_text(json.dumps(notebook_json,indent=2))
    metadata=dict(id='indarkarhana/biohub-trajectory-motion-acceptance',title='Biohub Trajectory Motion Acceptance',
                  code_file='trajectory-motion-acceptance.ipynb',language='python',kernel_type='notebook',
                  is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
                  dataset_sources=['indarkarhana/biohub-trajectory-motion-runtime-v1'],kernel_sources=[],
                  competition_sources=['biohub-cell-tracking-during-development'],model_sources=[],
                  docker_image='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
                  machine_shape='NvidiaTeslaT4')
    validate_submission_kernel_metadata(metadata)
    (notebook/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    receipt=dict(status='staged_not_launched',packaging_revision=revision,contract_sha256=contract_sha,
                 archive_sha256=archive_sha,archive_bytes=archive_path.stat().st_size,
                 files=len(pins)+1,wheels=len(wheels),notebook_sha256=sha(notebook/'trajectory-motion-acceptance.ipynb'),
                 dataset_private=True,internet_enabled=False,required_gpus=2,
                 acceptance_wall_cap_seconds=3600,production_not_staged=True,submission_performed=False)
    (ROOT/f'reports/experiments/trajectory-kaggle-runtime-v{revision}-build.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
