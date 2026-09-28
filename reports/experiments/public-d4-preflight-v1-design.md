# Public complete-D4 GPU preflight

Status: prepared and staged; no GPU execution. September 10, 2026 UTC.

One isolated geometric correction to the fixed LF-DCTTA public reference.
Run only after the user frees Antelume; do not start the instance implicitly
or disturb RSNA. Fresh read-only instance/GPU inventory before deployment.

- Single visible, idle GPU; refuse any existing compute PID, never kill it.
- Two CPU threads; PyTorch allocator limited to 45% of the GPU memory.
- 1,200-second internal watchdog; also use an outer Linux timeout at launch.
- Exact three weight hashes and source hashes in the staged manifest.
- Image input only: original cached full frames 46 and 47 of the previously
  exposed training-diagnostic movie 44b6_24264f12; no GEFF/labels/target movies.
- Execute reviewed encoder-only AST extracts, not the public notebook or its
  dependency installation, proxy sweep, graph processing or submission cells.
- Verify actual eight calls per encoder/DeepCenter arm, seven original versus
  eight corrected unique input permutations, finite outputs and genuine paired
  differences. Report CUDA memory and wall time, not synthetic GPU estimates.
- Store paired arrays and a terminal result. Verify all inputs unchanged.

Invocation inside an isolated Biohub preflight directory (bundle already
copied and hashes checked; existing project environments remain unchanged):

```text
timeout --signal=TERM --kill-after=15s 1220s <python-with-cuda> run-public-d4-preflight-v1.py --bundle . --output result --manifest-sha256 879906e3da6164493aa9a2a83bf27012f58c780557fa03e8702a0e7d4b10a1be
```

The staged runner owns its own timeout exit only. It does not automatically
install dependencies, train, launch a full run, stop the cloud instance or
submit to Kaggle. Host/network/staging time is not included in GPU inference
measurements. Full paired movie inference, the patched official scorer, embryo
and worst-movie checks, independence audit, offline/schema and hidden-runtime
budgeting remain necessary before promotion. Public checkpoint train overlap
must remain explicit; no automatic full-run authorization follows this smoke.
