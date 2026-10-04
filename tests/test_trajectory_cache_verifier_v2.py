from pathlib import Path
import runpy


def test_paired_platform_correction_retains_strict_identity_and_smoke_checks():
    root=Path(__file__).resolve().parents[1]
    source=runpy.run_path(str(root/'scripts/verify-trajectory-overlap-cache-v2.py'))['configured_source']()
    assert 'b5a86113ee182a0dc437201f10773f72f7370115d3b622d14e4640031d08d8ec' in source
    assert 'exact_paired_T4_graph_identity=identical' in source
    assert 'initial_A10_reference_check_failed=True' in source
    assert 'All eight movies required before truth access' in source
    assert 'Four-worker smoke and full acceptance required' in source
    assert source.index("raise ValueError('Cached graph differs from paired accepted T4 graph; no rescoring')") < source.index('    ground_truth_opened=False')
