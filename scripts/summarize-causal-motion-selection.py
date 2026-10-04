"""Verify causal cached-node scores against the archived native control."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare_static(causal, static):
    if (causal['checkpoint_sha256'] != static['checkpoint_sha256']
            or causal['per_movie']['original'] != static['per_movie']['original']
            or static['target_audit_opened'] is not False
            or static['authorized_for_submission'] is not False):
        raise ValueError('Static control must use exact same checkpoint and native graphs')
    rows = causal['per_movie']['motion']
    old_rows = static['per_movie']['motion']
    if [r['stem'] for r in rows] != [r['stem'] for r in old_rows]:
        raise ValueError('Static control movie coverage mismatch')
    for row, old in zip(rows, old_rows):
        if any(row[key] != old[key] for key in ('num_pred_nodes', 'node_recall', 'total_node_ratio')):
            raise ValueError('Static control detections changed')
        if static['receipts'][row['stem']]['nodes_unchanged'] is not True:
            raise ValueError('Static control coordinate receipt missing')
    return dict(static_summary=static['summaries']['motion'],
                causal_minus_static=causal['summaries']['motion']['score']-static['summaries']['motion']['score'],
                adjusted_edge_regressions=[r['stem'] for r, old in zip(rows, old_rows)
                    if r['adj_edge_jaccard'] < old['adj_edge_jaccard']-1e-12],
                caveat='Whole causal-prior package comparison: history, fit sample and variance differ; not a pure velocity-only ablation')


def summarize(result, control, fit):
    # Builder embeds read_text(UTF-8): Windows CRLF becomes LF in Linux.
    runtime_fit = fit.replace(b'\r\n', b'\n')
    if (result['checkpoint_sha256'] != control['checkpoint_sha256']
            or result['target_audit_opened'] is not False
            or result['authorized_for_submission'] is not False
            or result['fit_receipt_sha256'] != hashlib.sha256(runtime_fit).hexdigest()):
        raise ValueError('Wrong checkpoint, fit receipt, or scope')
    expected = control['per_movie']
    for arm in ('original', 'motion'):
        rows = result['per_movie'][arm]
        if [r['stem'] for r in rows] != [r['stem'] for r in expected]:
            raise ValueError('Incomplete ordered movie coverage')
        for row, old in zip(rows, expected):
            keys = old.keys() if arm == 'original' else ('num_pred_nodes', 'node_recall', 'total_node_ratio')
            for key in keys:
                if isinstance(old[key], (int, float)):
                    equal = math.isclose(row[key], old[key], rel_tol=0, abs_tol=1e-12)
                else:
                    equal = row[key] == old[key]
                if not equal:
                    raise ValueError('Native/detection evidence changed: '+key)
    for row in expected:
        if result['receipts'][row['stem']]['nodes_unchanged'] is not True:
            raise ValueError('Missing coordinate preservation receipt')
    delta = result['summaries']['motion']['score'] - control['summary']['score']
    if not math.isclose(delta, result['score_delta'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Reported score delta inconsistent')
    regressions = [r['stem'] for r, old in zip(result['per_movie']['motion'], expected)
                   if r['adj_edge_jaccard'] < old['adj_edge_jaccard'] - 1e-12]
    return dict(status='verified_causal_motion_selection', result=result, score_delta=delta,
                adjusted_edge_regressions=regressions, detection_metrics_identical=True,
                decision='reject' if delta <= 0 else 'selection_gain_requires_further_validation',
                authorized_for_submission=False)


if __name__ == '__main__':
    path = ROOT/'.biohub/cache/kernel-outputs/causal-motion-selection-v1/causal_motion_score/causal_motion_score.json'
    control = ROOT/'reports/experiments/independent-joint-selection-v1-score.json'
    fit = ROOT/'reports/experiments/independent-motion-persistence-training.json'
    report = summarize(json.loads(path.read_text()), json.loads(control.read_text())['result'], fit.read_bytes())
    report['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    static_path = ROOT/'.biohub/cache/kernel-outputs/joint-static-motion-selection-v1/joint_static_motion_score/static_motion_score.json'
    if static_path.is_file():
        report['static_control'] = compare_static(report['result'], json.loads(static_path.read_text()))
        report['static_source_sha256'] = hashlib.sha256(static_path.read_bytes()).hexdigest()
    (ROOT/'reports/experiments/causal-motion-selection-v1-score.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'score_delta', 'adjusted_edge_regressions', 'decision')}, indent=2))
    if 'static_control' in report:
        print(json.dumps(report['static_control'], indent=2))
