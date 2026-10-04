"""CPU-only frozen repair diagnostic. Build every repair before opening GT."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_edge_consensus import repair

M = runpy.run_path(str(ROOT / 'scripts/score-focus-raw-linker.py'))


def run(root):
    raw, receipt = M['validate'](root,
        ROOT / '.biohub/cache/kernel-outputs/focus3d-raw-detections-v1')
    controls = M['REPLAY']['validate_arms'](
        ROOT / '.biohub/cache/kernel-outputs/focus-bridge-cached-control-v2')
    contract_path = ROOT / 'research/focus_edge_consensus_v1_contract.json'
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    repaired, receipts = {}, {}
    for stem in sorted(raw):
        repaired[stem], receipts[stem] = repair(controls[stem]['control'], raw[stem])
        receipts[stem]['prediction_sha256'] = hashlib.sha256(
            json.dumps(repaired[stem], sort_keys=True).encode()).hexdigest()
        print(json.dumps(dict(event='prelabel_repair', stem=stem, **receipts[stem])), flush=True)
    # All repairs and hashes are frozen in memory before GT access.
    import tracksdata as td
    from geff import GeffMetadata
    scorer = M['REPLAY']['load_scorer']('current')
    rows = {'control': [], 'repair': []}
    for stem in sorted(raw):
        truth_path = ROOT / '.biohub/cache/competition-train-geffs-packed-v1/train' / (stem + '.geff')
        for arm, payload in [('control', controls[stem]['control']), ('repair', repaired[stem])]:
            graph = M['REPLAY']['prediction_graph'](payload)
            truth = td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er = scorer.evaluate(graph, truth, scale=(1.625, .40625, .40625), max_distance=7.)
            count = float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row = dict(scorer.per_sample_metrics(er, count, scorer.node_recall(graph, truth)),
                       stem=stem, embryo=stem.split('_')[0])
            rows[arm].append(row)
            print(json.dumps(dict(event='scored', arm=arm, **row)), flush=True)
    summaries = {k: scorer.summarise(v) for k, v in rows.items()}
    delta = summaries['repair']['score'] - summaries['control']['score']
    tp_delta = sum(r['edge_tp'] - c['edge_tp'] for c, r in zip(rows['control'], rows['repair']))
    fp_delta = sum(r['edge_fp'] - c['edge_fp'] for c, r in zip(rows['control'], rows['repair']))
    passed = (delta > 0 and tp_delta > 0 and fp_delta == 0
              and all(r['adj_edge_jaccard'] >= c['adj_edge_jaccard']
                      for c, r in zip(rows['control'], rows['repair'])))
    return dict(run_id='focus-edge-consensus-v1', per_movie=rows, summaries=summaries,
        by_embryo={arm: {e: scorer.summarise([r for r in values if r['embryo'] == e])
                        for e in ('44b6', '6bba')} for arm, values in rows.items()},
        receipts=receipts, contract_sha256=contract_hash,
        score_delta=delta, correct_link_delta=tp_delta, false_link_delta=fp_delta,
        diagnostic_gate_passed=passed, validation_scope='base-training diagnostic',
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        authorized_for_submission=False, authorized_for_production_promotion=False,
        gpu_hours=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2))
