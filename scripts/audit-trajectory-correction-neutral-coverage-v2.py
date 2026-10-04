"""Count source-only candidate FP-supervision gaps without fitting or new GT."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    folder = ROOT / '.biohub/cache/trajectory-correction-source-v1'
    dataset = read(folder / 'RESULT.json')
    assert sha(folder / 'RESULT.json') == 'e279e24348d17e60e07b8e6bb964ffab95490e502a3f70ac67ecd35928dfa1db'
    receipt = ROOT / 'reports/experiments/trajectory-correction-neutral-coverage-v2.json'
    assert not receipt.exists()
    by_movie, by_embryo = {}, {e: Counter() for e in ('44b6', '6bba')}
    for stem, item in dataset['records'].items():
        lp, cp = folder / (stem + '-labels.json'), folder / (stem + '-components.json')
        assert sha(lp) == item['labels_sha256'] and sha(cp) == item['components_sha256']
        labels, parts = read(lp), read(cp)
        assert len(labels) == len(parts) == item['counts']['components']
        counts = Counter()
        for label, part in zip(labels, parts):
            counts['total'] += 1
            counts['existing_positive' if label['label'] == 1 else
                   'existing_negative' if label['label'] == 0 else 'ignored'] += 1
            if label['label'] != -1 or not label.get('known_children', 0):
                continue
            counts['ignored_with_known_children'] += 1
            if len(part['removed']) == len(part['added']):
                continue
            counts['ignored_known_unequal_edge_counts'] += 1
            complete = not label.get('unknown_children', 0) and not label.get('ambiguous_children', 0)
            counts['fully_known_candidates' if complete else 'mixed_unknown_candidates'] += 1
            if complete:
                counts['fully_known_net_removal' if len(part['removed']) > len(part['added'])
                       else 'fully_known_net_addition'] += 1
        by_movie[stem] = dict(counts)
        by_embryo[item['embryo']].update(counts)
    result = dict(status='source_neutral_label_coverage_audited', source_only=True,
        new_ground_truth_opened=False, model_fitted=False, authorized_for_submission=False,
        counts_are_audit_candidates_not_verified_metric_labels=True,
        dataset_sha256=sha(folder / 'RESULT.json'), source_sha256=sha(Path(__file__)),
        movies=by_movie, by_embryo=by_embryo)
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], by_embryo=by_embryo)))


if __name__ == '__main__':
    main()
