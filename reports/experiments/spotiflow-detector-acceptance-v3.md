# spotiflow-detector-acceptance-v3

Status: completed in 220.303 seconds; drop-in pretrained detector retired.

V3 propagates that exact path through `PYTHONPATH` to the evaluator subprocess.
The pre-inference setup still preserves NumPy 2.0.2, imports SciPy and Spotiflow,
and loads a complete GEFF graph. Candidate selection and four-movie acceptance
are unchanged, and this run cannot create or submit `submission.csv`.

Resource envelope: one T4, Internet off, hard stop 6,900 seconds, declared
maximum two GPU hours.

The selection split chose official `smfish_3d` with Spotiflow auto
normalization (annotated recall 0.68884). On the four disjoint complete
acceptance movies it reached 0.77768 recall over 2,357 annotated nodes, versus
0.96903 for the frozen public detector: delta -0.19134, with a negative delta on
all four movies. This is decisive evidence against swapping the official model
directly into the pipeline.

The learned lane remains justified: these checkpoints were never adapted to
the Biohub image distribution. `smfish_3d` is retained only as the empirically
better warm start for corrected dense synthetic fine-tuning. The fine-tuned
checkpoint must pass this same disjoint acceptance protocol before candidate
assembly.
