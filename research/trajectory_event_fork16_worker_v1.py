"""Narrow worker adaptation matching the frozen learned fork16 evaluator."""
import ast
from research.trajectory_portable_adapter_v1 import replace_once

OLD_CALL = "event.refine(raw_ilp, repaired, coords, np.asarray(edges), event_model['weights'])"
NEW_CALL = "event.refine(raw_ilp, repaired, coords, np.asarray(edges), event_model['weights'], per_frame_seconds=10., max_seconds=600.)"
SCHEMA_MARKER = "    if tuple(event_model['features']) != event.FEATURES:"
FORK_GUARD = "    if event_model.get('max_fork_children') != 16:\n        raise ValueError('Expected learned fork16 model vocabulary')\n"


def adapt_worker(source):
    ast.parse(source)
    changed = replace_once(source, OLD_CALL, NEW_CALL)
    changed = replace_once(changed, SCHEMA_MARKER, FORK_GUARD + SCHEMA_MARKER)
    ast.parse(changed)
    assert changed.replace(NEW_CALL, OLD_CALL).replace(FORK_GUARD, '') == source
    return changed
