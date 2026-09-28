# Structured candidate runtime policy, before public outputs are opened

The frozen eight-movie two-T4 run completed in 952.0428 seconds. Its four
workers reported 3381.9718 aggregate movie-wall seconds. Effective concurrent
workers including startup and assembly are therefore 3.5523, on exactly two
GPUs. This is not a four-GPU run and not four independent uncontented workers.

The historical two-worker release verifier divided aggregate movie time by two.
Applied unchanged here, that would predict 11.68 hours for the same validation
run whose actual measured cohort projection is 6.5784 hours. Such incompatible
accounting must not be silently reused or described as measured throughput.

Revision 3 retains the ten-hour watchdog, 12-hour Kaggle limit, 20 GPU-hour
worst-case reservation, eight-hour quota reserve, and complete-movie quality
requirements. Before reading the public timing results, define:

- Validation makespan projection for the historical 199-movie workload must be
  at most eight hours, preserving two hours of initial planning headroom.
- Public average worker workload is converted using the measured validation
  concurrency, not assumed four-GPU capacity. It must project below ten hours.
- Pooled actual validation and public makespans, normalized over twelve movies,
  must also project below ten hours.
- Record the public-only makespan projection even when it is dominated by one
  dense movie. Do not conceal an adverse estimate or claim hidden runtime is
  guaranteed. Static-shard imbalance, different hidden density, and transfer of
  scheduling efficiency remain explicit risks that must be disclosed.

This is an engineering workload extrapolation, not a statistical bound or a
change to model selection. No model, predictions, thresholds, or scorer change.
If the frozen gates fail, preserve the failure and do not silently weaken them.
