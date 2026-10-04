from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import pytest
import torch
from bn_recalibration import recalibrate_batchnorm


def test_only_running_buffers_change_and_modes_restore():
    model = torch.nn.Sequential(torch.nn.Linear(1,1,bias=False),torch.nn.BatchNorm1d(1)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1.)
    weights = [p.clone() for p in model.parameters()]
    result = recalibrate_batchnorm(model,[torch.tensor([[1.],[3.]]),torch.tensor([[5.],[7.]])],model)
    assert result['parameters_unchanged'] and result['batches'] == 2
    assert torch.allclose(model[1].running_mean,torch.tensor([4.]))
    assert torch.allclose(model[1].running_var,torch.tensor([2.]))
    assert all(torch.equal(a,b) for a,b in zip(weights,model.parameters()))
    assert not model.training and not model[1].training and model[1].momentum == .1


def test_non_bn_buffer_mutation_rejected():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(1)).eval()
    model.register_buffer('other',torch.tensor(0))
    def encode(batch):
        model.other.add_(1)
        return model(batch)
    with pytest.raises(ValueError,match='Non-BatchNorm'):
        recalibrate_batchnorm(model,[torch.ones(2,1)],encode)


def test_empty_or_missing_bn_rejected():
    with pytest.raises(ValueError,match='No tracked'):
        recalibrate_batchnorm(torch.nn.Linear(1,1),[],lambda x:x)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(1)).eval()
    with pytest.raises(ValueError,match='Empty'):
        recalibrate_batchnorm(model,[],model)
    assert not model[0].training and model[0].momentum == .1
