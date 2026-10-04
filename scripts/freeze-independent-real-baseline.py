"""Print split provenance from directory names only; never open image/GT data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.independent_real_baseline import make_split

if __name__ == '__main__':
    root = ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train'
    stems = sorted(p.stem for p in root.glob('*.geff') if p.is_dir())
    if len(stems) != 199:
        raise ValueError('Expected complete 199-movie training inventory')
    result = make_split(stems)
    result['inventory_sha256'] = hashlib.sha256(('\n'.join(stems) + '\n').encode()).hexdigest()
    result['organizer_commit'] = '075fc5f5a52d11077f9dc2b074644618f26939e2'
    result['reciprocal_freeze_rule'] = 'Freeze both fold checkpoints and inference settings before opening either target-embryo audit result; never ensemble a model trained on that audit embryo'
    result['pilot'] = dict(status='not_launched', maximum_gpu_hours=1,
        architecture='Organizer TemporalUNet3D [32,64,128] and four-block association transformer',
        input='Real source-embryo consecutive frame pairs; no synthetic-only training',
        training_coordinates='Model detections matched to sparse annotations, not GT-only linker inputs',
        precision='FP16 autocast encoder with GradScaler; FP32 matching and sparse losses',
        promotion='No deployment promotion from a throughput/optimization pilot',
        followup='Continue only with finite improving real training losses, usable detection coverage, and measured time/VRAM budget')
    print(json.dumps(result, indent=2))
