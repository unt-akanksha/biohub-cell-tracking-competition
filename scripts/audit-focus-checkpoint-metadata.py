"""CPU-only restricted checkpoint metadata inspection; never constructs a model."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from collections.abc import Mapping

MODEL_SHA = 'b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a'


def describe_checkpoint(value, is_tensor):
    metadata, tensor_count, elements, truncated = {}, 0, 0, False
    def visit(item, path, depth):
        nonlocal tensor_count, elements, truncated
        if is_tensor(item):
            tensor_count += 1
            elements += item.numel()
        elif item is None or isinstance(item, (bool, int, float, str)):
            metadata[path] = item[:2000] if isinstance(item, str) else item
        elif depth >= 5:
            metadata[path] = {'type': type(item).__name__, 'depth_limit': True}
            truncated = True
        elif isinstance(item, Mapping):
            for index, (key, child) in enumerate(item.items()):
                if index >= 5000:
                    truncated = True
                    break
                visit(child, path + '/' + str(key), depth+1)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item[:100]):
                visit(child, path + '/' + str(index), depth+1)
            truncated |= len(item) > 100
        else:
            metadata[path] = {'type': type(item).__name__, 'unsupported_metadata_type': True}
    visit(value, '', 0)
    return dict(top_level_type=type(value).__name__, tensor_count=tensor_count,
                tensor_elements=elements, non_tensor_metadata=metadata, metadata_truncated=truncated)


def main():
    started = time.monotonic()
    target = Path('/kaggle/working/focus_checkpoint_metadata.json')
    def timeout():
        target.write_text(json.dumps(dict(status='timeout', gpu_hours=0)))
        os._exit(124)
    timer = threading.Timer(840, timeout)
    timer.daemon = True
    timer.start()
    result = dict(status='error', gpu_hours=0, competition_data_read=False,
                  model_constructed=False, authorized_for_submission=False)
    try:
        candidates = [Path('/kaggle/input') / p / 'models/model_final_nuclei.pth' for p in (
            'datasets/qiweiyin/focus3d-nuclei-runtime', 'qiweiyin/focus3d-nuclei-runtime', 'focus3d-nuclei-runtime')]
        paths = {p.resolve() for p in candidates if p.is_file()}
        if len(paths) != 1:
            raise ValueError('Exactly one explicit public checkpoint mount required')
        path = paths.pop()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != MODEL_SHA:
            raise ValueError('Author checkpoint identity mismatch')
        import torch
        torch.set_num_threads(2)
        # No unsafe-unpickling fallback, model construction, tensor computation or GPU.
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        result.update(status='inspected', model_sha256=MODEL_SHA, restricted_load=True,
                      torch_version=torch.__version__, **describe_checkpoint(checkpoint, torch.is_tensor))
    except Exception as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc)[:2000])
        raise
    finally:
        timer.cancel()
        result.update(elapsed_seconds=time.monotonic()-started, declared_budget_seconds=900,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        target.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
