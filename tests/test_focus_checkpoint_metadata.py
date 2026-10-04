from pathlib import Path
import runpy

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/audit-focus-checkpoint-metadata.py'))


class TensorStub:
    def numel(self):
        return 100


def test_metadata_kept_and_tensor_values_never_read():
    checkpoint = dict(model={'weight': TensorStub()}, iteration=42, dataset=['known_train'], config=dict(seed=1))
    result = M['describe_checkpoint'](checkpoint, lambda x: isinstance(x, TensorStub))
    assert result['tensor_count'] == 1 and result['tensor_elements'] == 100
    assert result['non_tensor_metadata'] == {'/iteration': 42, '/dataset/0': 'known_train', '/config/seed': 1}
    assert result['metadata_truncated'] is False


def test_pure_state_dictionary_has_no_training_provenance():
    result = M['describe_checkpoint']({'a': TensorStub(), 'b': TensorStub()}, lambda x: isinstance(x, TensorStub))
    assert result['tensor_count'] == 2 and result['non_tensor_metadata'] == {}


def test_metadata_limits_are_explicit():
    result = M['describe_checkpoint']({'values': list(range(101))}, lambda x: False)
    assert result['metadata_truncated']
    assert len(result['non_tensor_metadata']) == 100
