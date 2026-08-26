# spotiflow-detector-acceptance-v3

Status: staged after the v2 environment passed but its evaluator subprocess did
not inherit the support-pack `biohub_tracking` source path.

V3 propagates that exact path through `PYTHONPATH` to the evaluator subprocess.
The pre-inference setup still preserves NumPy 2.0.2, imports SciPy and Spotiflow,
and loads a complete GEFF graph. Candidate selection and four-movie acceptance
are unchanged, and this run cannot create or submit `submission.csv`.

Resource envelope: one T4, Internet off, hard stop 6,900 seconds, declared
maximum two GPU hours.
