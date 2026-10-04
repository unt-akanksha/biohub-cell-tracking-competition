"""Stage a pinned source batch's unchanged image pipeline after image download."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text, old, new):
    assert text.count(old) == 1
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    report = json.loads((ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').read_text())
    record = next(b for b in report['batches'] if b['index'] == args.batch)
    scope = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(args.batch))
    assert sha(scope / 'MOVIES.json') == record['plan_sha256']
    plan = json.loads((scope / 'MOVIES.json').read_text())
    stems = [m['stem'] for m in plan['movies']]
    image_path = scope / 'IMAGE_MANIFEST.json'
    images = json.loads(image_path.read_text())
    assert images['status'] == 'complete' and images['stems'] == stems
    assert len(images['records']) == 102 * len(stems) and not images['ground_truth_opened']
    assert images['private_plan_sha256'] == sha(scope / 'PRIVATE_ARCHIVE_PLAN.json')
    assert images['bytes'] <= record['maximum_image_cache_bytes']
    base = ROOT / '.biohub/cache/trajectory-division-full-movie-v1-bundle'
    assert sha(base / 'CONTRACT.json') == '7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1'
    contract = json.loads((base / 'CONTRACT.json').read_text())
    for name, digest in contract['bundle_sha256'].items():
        assert sha(base / name) == digest
    target = ROOT / '.biohub/cache' / ('trajectory-event-source-v1-b' + str(args.batch) + '-bundle')
    target.mkdir(exist_ok=False)
    for name in contract['bundle_sha256']:
        shutil.copy2(base / name, target / name)
    helper = target / 'trajectory_division_full_movie_v1.py'
    source = replace(helper.read_text(encoding='utf-8'),
        "STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')", 'STEMS = ' + repr(tuple(stems)))
    compile(source, str(helper), 'exec')
    helper.write_bytes(source.encode())
    runner = target / 'run-trajectory-division-full-movie-v1.py'
    source = replace(runner.read_text(encoding='utf-8'), "len(image_manifest['records']) != 408",
                     "len(image_manifest['records']) != " + str(102 * len(stems)))
    source = replace(source, "run_id='trajectory-division-full-movie-v1'", 'run_id=' + repr(plan['run_id']))
    source = replace(source, "cap = 600 if args.mode == 'smoke' else 3600", "cap = 600 if args.mode == 'smoke' else 2700")
    source = replace(source, '    def timeout():\n',
        "    def timeout():\n        result.update(status='timeout', elapsed_seconds=time.monotonic()-started)\n        (args.output / 'result.json').write_text(json.dumps(result, indent=2))\n")
    compile(source, str(runner), 'exec')
    runner.write_bytes(source.encode())
    contract.update(run_id=plan['run_id'], stems=stems, image_manifest_sha256=sha(image_path),
                    source_scope_sha256=sha(scope / 'MOVIES.json'), source_predictions_only=True,
                    model_training_performed=False, ground_truth_included=False,
                    purpose='Raw, ILP, public-postprocess and AR2 source graphs for future structured event fitting')
    contract['full'].update(movies=len(stems), wall_cap_seconds=2700)
    contract['bundle_sha256'] = {name: sha(target / name) for name in contract['bundle_sha256']}
    (target / 'CONTRACT.json').write_bytes((json.dumps(contract, indent=2) + '\n').encode())
    archive = target.with_suffix('.tar')
    with tarfile.open(archive, 'w') as tar:
        tar.add(target, arcname=target.name)
    receipt = dict(status='source_collection_staged_not_launched', batch=args.batch,
                   contract_sha256=sha(target / 'CONTRACT.json'), archive_sha256=sha(archive),
                   images_sha256=sha(image_path), archive_bytes=archive.stat().st_size,
                   movies=stems, full_wall_cap_seconds=2700, smoke_required=True,
                   source_scope_sha256=sha(scope / 'MOVIES.json'), authorized_for_submission=False)
    (ROOT / 'reports/experiments' / ('trajectory-event-source-v1-b' + str(args.batch) + '-build.json')).write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
