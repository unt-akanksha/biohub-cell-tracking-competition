from __future__ import annotations

from research.spotiflow_biohub.evaluate_pu_detector import summarize_rows


def test_summary_pools_by_annotated_node_count_and_prefix() -> None:
    rows = [
        {
            "stem": "44b6_a",
            "candidate": {
                "annotated_gt_nodes": 10,
                "matched_gt_nodes": 9,
                "annotated_node_recall": 0.9,
            },
        },
        {
            "stem": "6bba_b",
            "candidate": {
                "annotated_gt_nodes": 30,
                "matched_gt_nodes": 24,
                "annotated_node_recall": 0.8,
            },
        },
    ]
    summary = summarize_rows(rows)
    assert summary["annotated_node_recall"] == 33 / 40
    assert summary["worst_movie_recall"] == 0.8
    assert summary["by_prefix"]["44b6"]["annotated_node_recall"] == 0.9
    assert summary["by_prefix"]["6bba"]["annotated_node_recall"] == 0.8
