"""Freeze source, head protocol and merged data identity before experiments."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--additive-v5',action='store_true');args=parser.parse_args()
    bundle_name='native-division-additive-v5-head' if args.additive_v5 else 'native-division-v3-head'
    output = ROOT / '.biohub/cache' / (bundle_name+'-bundle')
    output.mkdir(exist_ok=False)
    files = ['scripts/extract-division-features-v3.py', 'scripts/train-division-heads-v3.py',
             'research/native_correspondence_models_v2.py', 'research/visual_correspondence_models_v1.py',
             'research/native_division_features_v3.py', 'tests/test_native_division_features_v3.py',
             'reports/experiments/native-division-enrichment-v3-design.md']
    if args.additive_v5:files+=['scripts/merge-division-features-v5.py','reports/experiments/native-division-additive-v5-design.md']
    for name in files:
        destination = output / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    contract = dict(run_id='native-division-v3-heads', feature_max_seconds=600, head_max_seconds=300,
                    data_sha256='7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06',
                    files=[dict(path=p.relative_to(output).as_posix(), sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()],
                    authorized_for_submission=False, target_pilot_labels_used=False)
    if args.additive_v5:
        data=ROOT/'.biohub/cache/native-division-additive-v5-full-output/DATA.json'
        if json.loads(data.read_text())['status']!='additive_extraction_complete':raise ValueError('New positive extraction incomplete')
        contract.update(run_id='native-division-additive-v5-heads',data_sha256=sha(data))
    (output / 'CONTRACT.json').write_text(json.dumps(contract, indent=2) + '\n')
    archive = output.with_suffix('.tar')
    with tarfile.open(archive, 'w') as tar:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(output).as_posix())
    receipt = dict(status='division_heads_packaged', archive_sha256=sha(archive),
                   contract_sha256=sha(output / 'CONTRACT.json'), bytes=archive.stat().st_size)
    (ROOT / 'reports/experiments' / (bundle_name+'-build.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
