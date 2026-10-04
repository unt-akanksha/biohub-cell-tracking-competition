from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from edge_feature_tta import D4_VIEWS, transform, invert, install_edge_feature_tta


def test_eight_unique_views_and_exact_inverses_on_rectangular_grid():
    x = torch.arange(2*3*4*5).reshape(2,3,4,5)
    views = []
    for view in D4_VIEWS:
        y = transform(x,*view)
        assert torch.equal(invert(y,*view),x)
        views.append((tuple(y.shape),tuple(y.flatten().tolist())))
    assert len(set(views)) == 8


class Encoder(torch.nn.Module):
    def __init__(self, position_dependent=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.position_dependent = position_dependent
        self.calls = []

    def encode(self, images):
        features = self.weight * images.unsqueeze(2)
        if self.position_dependent:
            features = features + torch.arange(images.shape[-1],dtype=images.dtype)
        logits = [images[:,i].unsqueeze(1)*3. for i in range(images.shape[1])]
        self.calls.append((features.clone(), logits))
        return features, logits


@pytest.mark.parametrize('position_dependent',[False,True])
def test_features_average_but_detection_logits_and_weights_are_native(position_dependent):
    model = Encoder(position_dependent).eval()
    initial = model.weight.detach().clone()
    receipt = install_edge_feature_tta(model)
    features, logits = model.encode(torch.arange(2*3*4*5,dtype=torch.float32).reshape(1,2,3,4,5))
    expected = torch.stack([invert(row[0],*view) for row,view in zip(model.calls,D4_VIEWS)]).mean(0)
    torch.testing.assert_close(features,expected)
    assert logits is model.calls[0][1]
    assert torch.equal(model.weight,initial) and model.weight.grad is None
    assert len(model.calls) == 8 and receipt['encode_calls'] == 1
    assert (receipt['maximum_mean_absolute_feature_delta'] > 0) == position_dependent
    assert not features.requires_grad


def test_training_mode_or_double_install_rejected():
    model = Encoder()
    with pytest.raises(ValueError,match='evaluation'):
        install_edge_feature_tta(model)
    model.eval()
    install_edge_feature_tta(model)
    with pytest.raises(ValueError,match='already'):
        install_edge_feature_tta(model)
    model.train()
    with pytest.raises(ValueError,match='training'):
        model.encode(torch.zeros(1,2,3,4,5))
