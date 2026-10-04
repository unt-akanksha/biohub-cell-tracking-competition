import pytest

torch = pytest.importorskip('torch')
from research.native_division_features_v3 import joint_features, patch_statistics


def test_geometry_and_daughter_order():
    generator = torch.Generator().manual_seed(73)
    embeddings = [torch.randn(3, 256, generator=generator) for _ in range(2)]
    statistics = torch.rand(3, 6, generator=generator)
    triples = torch.tensor([[0, 1, 2]])
    coords = torch.tensor([[[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]])
    features = joint_features(embeddings, statistics, triples, coords)
    swapped = joint_features(embeddings, statistics, triples[:, [0, 2, 1]], coords[:, [0, 2, 1]])
    assert features.shape == (1, 2590)
    torch.testing.assert_close(features, swapped, rtol=0, atol=0)
    torch.testing.assert_close(features[0, :6], torch.tensor([.3, .4, .5, .25, 0., .1]))


def test_constant_statistics_and_empty():
    stats = patch_statistics(torch.ones(3, 3, 15, 15, 15))
    torch.testing.assert_close(stats, torch.ones(3, 6))
    result = joint_features([torch.zeros(3, 256)], stats, torch.zeros((0, 3), dtype=torch.long), torch.zeros(0, 3, 3))
    assert result.shape == (0, 1310)


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match='Nonfinite'):
        joint_features([], torch.full((3, 6), float('nan')), torch.tensor([[0, 1, 2]]), torch.zeros(1, 3, 3))
