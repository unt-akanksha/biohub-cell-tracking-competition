# Nonlinear presence model: feasibility FAIL

CPU-only experiment completed8.406s including verification and fitting worker.
One fixed100-tree depth2 model, lr0.05, min leaf20, no parameter sweep or early
stopping. Fitting worker received only2,870 training examples, not diagnostics.
Standalone presence probability uses the same seven context features plus
original presence log-odds; real-parent ranking and raw nodes remain frozen.

| Diagnostic arm | NLL | Correct parent /2645 | Correct absent /27 |
| --- | ---: | ---: | ---: |
| Physical-only | 0.333915 | 2500 | 18 |
| Original neural + physical | 0.219009 | 2585 | 12 |
| Previous linear presence | 0.193443 | 2589 | 12 |
| Shallow boosted presence | 0.194705 | 2590 | 8 |

Same frozen gate FAILS missing-parent correctness. Do not extend to full-source
tracking or retune based on this diagnostic. No source/target access, prediction
promotion or submission. These remain original-training-domain diagnostics,
not independent validation or official tracking scores.

Portable JSON modelSHA256:
`5f53282a3df99257db96c71872efa7c5e03ef91fca46725f42639d08412c9c14`.
Native sklearn1.9.0 versus exported real fitting logits max error8.88e-16;
synthetic native/export/JSON reload passed exactly. Float32 threshold boundary
and role-guard tests pass. No GPU used; reusable summaries are unchanged.

The absence deficit persists under linear and shallow nonlinear presence
models. This does not prove that all nonlinear models fail, but provides no
support for promoting these models or simply sweeping more settings on the
same small diagnostic. Further interventions need a distinct rationale and
their own frozen design, retaining the original full-movie promotion gates.
