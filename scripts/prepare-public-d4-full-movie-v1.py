"""Freeze the small supplemental runtime before any paired tracking scores."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import public_config, sha, STEMS, NOTEBOOK_SHA


def main():
    output = ROOT / '.biohub/cache/public-d4-full-movie-v1-bundle'
    output.mkdir(parents=True, exist_ok=False)
    notebook = ROOT / '.biohub/cache/public-frontier-20260910-2132/lf-dctta/biohub-lf-dctta.ipynb'
    config = public_config(notebook)
    (output / 'resolved-public-config.json').write_text(json.dumps(config, indent=2)+'\n')
    for source in ('research/public_d4_full_movie.py', 'scripts/run-public-d4-full-movie-v1.py'):
        shutil.copyfile(ROOT / source, output / Path(source).name)
    prior_dir = ROOT / '.biohub/cache/public-d4-preflight-v1'
    prior = json.loads((prior_dir / 'MANIFEST.json').read_text())
    pins = {name: row['sha256'] for name, row in prior['files'].items()
            if name not in ('run-public-d4-preflight-v1.py', 'frames.npz')}
    for name, digest in pins.items():
        if sha(prior_dir / name) != digest:
            raise ValueError('Existing preflight source changed')
    pins.update({path.name: sha(path) for path in output.iterdir()})
    contract = dict(run_id='public-d4-full-movie-v1', public_source_sha256=NOTEBOOK_SHA,
        bundle_sha256=pins, stems=list(STEMS),
        image_manifest_sha256=sha(ROOT / '.biohub/cache/public-d4-movie-download-v1/IMAGE_MANIFEST.json'),
        frames_per_movie=100, arms=['original','corrected'],
        scientific_delta='Only six predictor D4 rotation arguments and two DeepCenter arguments',
        execution_adapters=['AST definition extraction', 'strict image-only metadata loader',
                            'owned output directory', 'same models/settings, one CUDA GPU',
                            'CSV-equivalent integer coordinate serialization'],
        source_checkpoint_training_overlap=True, independently_held_out=False,
        smoke=dict(frames=8, movies=1, wall_cap_seconds=600),
        full=dict(movies=4, arms=2, wall_cap_seconds=3600),
        gpu='Antelume A10G only, idle compute PID check, sequential, 70% CUDA allocator limit',
        gate=dict(pooled_combined_strictly_improves=True, pooled_raw_edge_jaccard_strictly_improves=True,
                  per_embryo_combined_nonregression=True, per_movie_combined_nonregression=True,
                  worst_movie_analysis_required=True, all_movies_complete=True,
                  no_submission_promotion_from_overlapping_training_diagnostics=True),
        forbidden=['ground truth during inference','public predictions','metric exploits',
                   'public notebook proxy sweeps','leaderboard parameter selection'],
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        authorized_for_submission=False)
    path = output / 'CONTRACT.json'
    path.write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps(dict(status='staged', contract_sha256=sha(path), supplement_bytes=sum(
        p.stat().st_size for p in output.iterdir()), independently_held_out=False,
        authorized_for_submission=False)))


if __name__ == '__main__':
    main()
