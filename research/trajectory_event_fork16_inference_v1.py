"""Research-only uniform fork16 inference; the submitted fork8 code is untouched."""
import hashlib
from pathlib import Path

_base = Path(__file__).with_name('trajectory_event_pruned_inference_v1.py')
if hashlib.sha256(_base.read_bytes()).hexdigest() != '281d42bcc9bb01dcc7695b765924d5c956e9094d05d6cdfcec105ef9e08d6b9d':
    raise ValueError('Frozen inference implementation changed')
_source = _base.read_text(encoding='utf-8')
_old = 'for case in frames(initial,graph,groups):'
if _source.count(_old) != 1:
    raise ValueError('Unexpected frozen inference loop')
_source = _source.replace(_old, 'for case in frames(initial,graph,groups,max_fork_children=16):')
_namespace = dict(__name__=__name__ + '.frozen', __file__=str(_base))
exec(compile(_source, str(_base), 'exec'), _namespace)
refine = _namespace['refine']
FEATURES = _namespace['FEATURES']
MAX_FORK_CHILDREN = 16
