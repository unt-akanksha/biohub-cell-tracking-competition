from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from owned_detector_ensemble import probability_mixture_logit, install_owned_detector_ensemble


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_probability_identity_extremes_and_order(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    a = torch.linspace(-10, 10, 101, device=device)
    b = a.flip(0)*.7
    result = probability_mixture_logit(a, b)
    torch.testing.assert_close(result.sigmoid(), (a.sigmoid()+b.sigmoid())/2)
    extreme = torch.tensor([-1000., -100., 0., 100., 1000.], device=device)
    torch.testing.assert_close(probability_mixture_logit(extreme, extreme), extreme)
    peaks = probability_mixture_logit(torch.tensor([100., 101.], device=device),
                                      torch.tensor([100., 101.], device=device))
    assert peaks[1] > peaks[0] and torch.isfinite(peaks).all()
    with pytest.raises(ValueError): probability_mixture_logit(a, b[:1])
    with pytest.raises(ValueError): probability_mixture_logit(a, a*float('nan'))
    if device == 'cuda':
        with pytest.raises(ValueError): probability_mixture_logit(a, b.cpu())


class Detector(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        return images, [images[:, i:i+1]+self.bias for i in range(2)]


def test_ensemble_preserves_features_and_averages_each_d4():
    parent, second = Detector(0).eval(), Detector(2).eval()
    receipt = install_owned_detector_ensemble(parent, second)
    images = torch.arange(120.).reshape(1, 2, 3, 4, 5)/20
    features, logits = parent.encode(images)
    assert features is images and parent.calls == second.calls == 8
    assert not parent.bias.requires_grad and not second.bias.requires_grad
    for i, value in enumerate(logits):
        torch.testing.assert_close(value.sigmoid(),
            (images[:, i:i+1].sigmoid()+(images[:, i:i+1]+2).sigmoid())/2)
    assert receipt['encode_calls'] == 1 and receipt['maximum_mean_absolute_logit_delta'] > 0
    with pytest.raises(ValueError): install_owned_detector_ensemble(parent, second)
    second.train()
    with pytest.raises(ValueError): parent.encode(images)


def test_invalid_training_or_same_model():
    with pytest.raises(ValueError): install_owned_detector_ensemble(Detector(0), Detector(1))
    model = Detector(0).eval()
    with pytest.raises(ValueError): install_owned_detector_ensemble(model, model)
