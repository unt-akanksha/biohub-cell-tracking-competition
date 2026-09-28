"""Research adapter: identical frozen solver/inference code, vectorized pruning.

Private function namespaces prevent changes to the imported reference modules.
Not installed in the submitted runtime or the ongoing source fits.
"""
from types import FunctionType

from research import trajectory_event_dominance_v1 as dominance
from research.trajectory_event_fork16_inference_v1 import refine as reference_refine
from research.trajectory_event_vectorized_dominance_v1 import allowed_options


def private_function(function,replacements):
    namespace=dict(function.__globals__);namespace.update(replacements)
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=dict(function.__kwdefaults__ or {})
    return result


infer=private_function(dominance.infer,{'allowed_options':allowed_options})
refine=private_function(reference_refine,{'infer':infer})
