"""Freeze a no-truth GPU bundle for source-mixture division-positive validation."""
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_division_full_movie_v1 import STEMS


def main():
    output = ROOT / '.biohub/cache/trajectory-division-full-movie-v1-bundle'
    if output.exists():
        raise ValueError('Preserve frozen bundle')
    prior_root = ROOT / '.biohub/cache/public-d4-preflight-v1'
    prior = json.loads((ROOT / '.biohub/cache/public-d4-full-movie-v1-bundle/CONTRACT.json').read_text())
    models = ROOT / '.biohub/cache/learned-trajectory-endpoint-v1-models'
    if sha(models / 'manifest.json') != '57236f97756e7287204c1db32b084e0bbbe48315012b3117af444ddfa7296aa3':
        raise ValueError('Frozen expert manifest changed')
    model_manifest = json.loads((models / 'manifest.json').read_text())
    image_path = ROOT / '.biohub/cache/trajectory-division-archive-plan-v1/IMAGE_MANIFEST.json'
    images = json.loads(image_path.read_text())
    if images['status'] != 'complete' or images['stems'] != list(STEMS) or len(images['records']) != 408:
        raise ValueError('Complete image transfer required')
    output.mkdir(parents=True)
    for name, digest in prior['bundle_sha256'].items():
        if name in ('public_d4_full_movie.py', 'run-public-d4-full-movie-v1.py', 'resolved-public-config.json'):
            continue
        if sha(prior_root / name) != digest:
            raise ValueError('Pinned public runtime changed')
        shutil.copyfile(prior_root / name, output / name)
    config = ROOT / '.biohub/cache/public-d4-full-movie-v1-bundle/resolved-public-config.json'
    if sha(config) != prior['bundle_sha256']['resolved-public-config.json']:
        raise ValueError('Public inference configuration changed')
    shutil.copyfile(config, output / config.name)
    for record in model_manifest['models'].values():
        if sha(models / record['path']) != record['sha256']:
            raise ValueError('Frozen expert changed')
        if set(STEMS) & set(record['optimization_stems'] + record['calibration_stems']):
            raise ValueError('Division validation movie in motion fit')
        shutil.copyfile(models / record['path'], output / record['path'])
    for relative in ('research/trajectory_division_full_movie_v1.py',
                     'research/learned_trajectory_endpoint_v1.py',
                     'research/trajectory_source_mixture_v1.py',
                     'scripts/run-trajectory-division-full-movie-v1.py'):
        shutil.copyfile(ROOT / relative, output / Path(relative).name)
    pins = {p.name: sha(p) for p in output.iterdir()}
    contract = dict(run_id='trajectory-division-full-movie-v1', bundle_sha256=pins,
                    stems=list(STEMS), image_manifest_sha256=sha(image_path),
                    source_model_manifest_sha256=sha(models / 'manifest.json'),
                    mixture_design_sha256=sha(ROOT / 'reports/experiments/trajectory-source-mixture-v1-design.md'),
                    frames_per_movie=100, arms=['original'], repaired_variant='fixed_two_source_motion_support_union',
                    independently_held_out=False, public_backbone_training_overlap=True,
                    motion_model_validation_movies_excluded=True,
                    authorized_for_submission=False, ground_truth_included=False,
                    smoke=dict(frames=8, movies=1, wall_cap_seconds=600),
                    full=dict(frames=100, movies=4, wall_cap_seconds=3600),
                    gpu='Initially idle Antelume A10G; sequential,70% allocator ceiling; no RSNA changes')
    path = output / 'CONTRACT.json'; path.write_text(json.dumps(contract, indent=2) + '\n')
    archive = output.with_suffix('.tar.gz')
    if archive.exists():
        raise ValueError('Preserve staged archive')
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(output.iterdir()):
            tar.add(p, arcname=p.name, recursive=False)
    print(json.dumps(dict(status='staged', contract_sha256=sha(path), archive_sha256=sha(archive),
                         archive_bytes=archive.stat().st_size, files=len(pins)+1)), flush=True)


if __name__ == '__main__':
    main()
