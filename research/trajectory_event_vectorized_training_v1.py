"""Research-only exact hinge adapter; ongoing fit helpers are untouched."""
from research.trajectory_event_fast_training_v1 import hinge as reference_hinge
from research.trajectory_event_vectorized_inference_v1 import private_function
from research.trajectory_event_vectorized_dominance_v1 import allowed_options

hinge=private_function(reference_hinge,{'allowed_options':allowed_options})
