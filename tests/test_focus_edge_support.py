from pathlib import Path
import runpy

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/score-focus-raw-linker.py'))


def test_edge_support_partitions_missing_links_and_deduplicates_matches():
    support = M['edge_support']({0: 10, 1: 11, 2: 12, 3: None, 4: -1, 5: 10},
        [(0, 1), (5, 1), (1, 0), (3, 2), (4, 2)], {(10, 11), (11, 12), (12, 13)})
    assert support['correct'] == {(10, 11)}
    assert support['missing_despite_endpoints'] == {(11, 12)}
    assert support['missing_endpoint'] == {(12, 13)}


def test_complementarity_is_counts_only_and_does_not_mutate_inputs():
    truth = {(10, 11), (11, 12)}
    control = M['edge_support']({0: 10, 1: 11, 2: 12}, [(0, 1)], truth)
    candidate = M['edge_support']({0: 10, 1: 11, 2: 12}, [(1, 2)], truth)
    result = M['compare_support'](control, candidate)
    assert result['candidate_only_correct'] == result['control_only_correct'] == 1
    assert result['common_correct'] == 0
    assert result['diagnostic_oracle_union_correct'] == 2
    assert control['correct'] == {(10, 11)}
    assert all(isinstance(v, int) for v in result.values())
