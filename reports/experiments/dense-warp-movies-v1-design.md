# Fixed checkpoint complete-movie diagnostic

The partial screen failed its frozen ranking gate. It also revealed an
unattainable strict-gain requirement on the already-perfect 18-link 44b6 subset.
Do not erase those results or retrain/tune around them. This prospectively fixed
diagnostic asks a different downstream question: does changed confidence improve
the actual emitted graph under the unchanged competition pipeline?

Use four already exposed 100-frame movies whose raw data and original candidate
outputs are verified and resident: 44b6_12dfb391, 44b6_267148e4, 6bba_062c8d37,
6bba_07e24132. These were excluded from dense-warp optimization but have been
used in earlier unrelated diagnostics; the public backbone overlaps competition
training. This is not fresh CV or independent embryo-held-out model evidence.

Three fixed arms, all complete: original submitted pipeline, source44 linker,
source6 linker. Replace ONLY primary transformer tensors with the final 2,000-step
weights. Require all detector/encoder tensors exact, and raw detection arrays
identical to the control on every movie. Keep secondary model, original duplicate
D4 view, thresholds, low-margin fusion, ILP, DeepCenter, smoothing and trained
trajectory repair unchanged. Original control must reproduce its previously
validated output graph, ignoring only irrelevant edge-list order.

First run all three arms on eight frames of the first movie, with no labels,
strict loading, fixed raw detections, finite outputs and CSV-equivalent graph
checks. Only this same-contract smoke can launch the full run. Full run is
sequential Antelume only, 3 x 4 movies, estimated 30-40 minutes from previous
642.87-second four-movie control timing; hard limit 60 minutes. Use only new
Biohub RAM-backed output paths; no foreign-process interruption or Kaggle GPU.

After all predictions are frozen, run the patched official scorer on complete
movies. Each learned arm must strictly improve pooled combined score and raw
edge Jaccard, with no embryo or movie losing combined score, and worst-movie
reporting. Report both arms, not just the better one. No parameter sweep or
ensemble follows a failed arm. Passing this diagnostic still requires the
remaining four complete movies, broader confirmation and offline runtime before
any submission decision. Lower conditional loss alone never authorizes promotion.
