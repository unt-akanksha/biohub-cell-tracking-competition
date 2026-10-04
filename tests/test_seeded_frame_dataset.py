from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import numpy as np
import torch
from seeded_frame_dataset import seed_dataset


class ToyDataset:
    def __getitem__(self,index):
        rng = np.random.default_rng()
        return dict(imgs=torch.tensor(rng.random(5)),coords=torch.tensor([index]),masks=torch.tensor([True]))


def test_independent_streams_produce_identical_inputs_and_hashes():
    official = SimpleNamespace(np=np,torch=torch)
    left = seed_dataset(official,ToyDataset(),123)
    right = seed_dataset(official,ToyDataset(),123)
    for index in range(3):
        assert torch.equal(left[index]['imgs'],right[index]['imgs'])
    assert left.sample_hashes == right.sample_hashes and len(left.sample_hashes) == 3


def test_augmentation_state_can_be_restored():
    import copy
    dataset = seed_dataset(SimpleNamespace(np=np,torch=torch),ToyDataset(),7)
    state = copy.deepcopy(dataset.augmentation_rng.bit_generator.state)
    first = dataset[0]['imgs']
    dataset.augmentation_rng.bit_generator.state = state
    assert torch.equal(first,dataset[0]['imgs'])
