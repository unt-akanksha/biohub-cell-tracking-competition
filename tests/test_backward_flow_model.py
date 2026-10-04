from pathlib import Path
import io
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from backward_flow_model import BackwardFlowNet,observed_targets
from backward_flow_ops import sparse_backward_loss


def test_network_shape_learning_and_strict_reload():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = BackwardFlowNet()
    pair = torch.randn(1,2,17,19,21)
    before = model(pair)
    assert before.shape == (1,3,17,19,21) and before.abs().sum() == 0
    points = torch.tensor([[[8.,9.,10.]]])
    target = torch.tensor([[[.4,-.2,.1]]])
    mask = torch.tensor([[True]])
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    initial = sparse_backward_loss(before,points,target,mask).item()
    for _ in range(4):
        optimizer.zero_grad()
        loss = sparse_backward_loss(model(pair),points,target,mask)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        after = model(pair)
        assert sparse_backward_loss(after,points,target,mask).item() < initial
    buffer = io.BytesIO()
    torch.save(model.state_dict(),buffer)
    buffer.seek(0)
    restored = BackwardFlowNet().eval()
    restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(after,restored(pair),rtol=0,atol=0)


def test_parent_minus_child_units_division_and_unknown():
    coords = torch.tensor([[[[2.,2.,2.],[0.,0.,0.],[0.,0.,0.]],
                            [[2.,2.,3.],[2.,2.,1.],[1.,1.,1.]]]])
    masks = torch.tensor([[[True,False,False],[True,True,True]]])
    edges = torch.zeros(1,1,3,3)
    edges[0,0,0,:2] = 1
    points,target,observed = observed_targets(coords,masks,edges,torch.tensor([[2.,.5,.25]]))
    assert observed.tolist() == [[True,True,False]]
    torch.testing.assert_close(target[0,:2],torch.tensor([[0.,0.,-.25],[0.,0.,.25]]))
    torch.testing.assert_close(points,coords[:,1])
    masks[:,0] = False
    assert not observed_targets(coords,masks,edges,torch.ones(1,3))[2].any()


def test_merges_invalid_masks_and_units_rejected():
    coords = torch.zeros(1,2,2,3)
    masks = torch.ones(1,2,2,dtype=torch.bool)
    edges = torch.zeros(1,1,2,2)
    edges[0,0,:,0] = 1
    with pytest.raises(ValueError,match='merges'):
        observed_targets(coords,masks,edges,torch.ones(1,3))
    with pytest.raises(ValueError):
        observed_targets(coords,masks.float(),edges,torch.ones(1,3))
    with pytest.raises(ValueError,match='positive'):
        observed_targets(coords,masks,edges,torch.zeros(1,3))
