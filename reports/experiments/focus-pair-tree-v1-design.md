# Nonlinear residual candidate ranking: prospective functionality stage

The full72D linear appearance correction completed all12 folds but fails the
unchanged missing-parent requirement (84/161 versus115); LDA also fails84.
Neither is a candidate. Previous presence-only linear/tree/quadratic changes
are retired; this experiment changes real-candidate ranking through appearance
and motion interactions, not just presence or a threshold.

Use one fixed depth3 histogram-tree residual on top of the full72D linear
candidate logits.100trees, learning rate0.05, max_bin64, min_child_weight1,
lambda1, alpha0, max_delta_step1, no row/column sampling, seed244691,2CPUthreads.
Class weights remain1parent/sqrt(Nparent/Nabsent)absence. All original candidate
and null rows enter a grouped softmax objective, not independent binary labels.
Unknown targets remain excluded from fitting. Features are the unchanged72D
representation cast tofloat32 for tree branching; original linear margins and
loss arithmetic remainfloat64 externally. XGBoost learns only the residual,
with base_score0; never silently drop or double-add the external baseline.

The exact group gradient is used with diagonal Hessian upper bound2wp(1-p),
floor1e-12 for numerical stability. For each group D-H is the sum overi<j of
wp_i p_j(e_i+e_j)(e_i+e_j)^T, hence positive semidefinite. This is our adaptation
of the documented diagonal-bound approach, not an official supported custom
ranking objective. No qid/LambdaRank or independent-row cross-entropy is used.
Test derivatives, the bound and full-group shift invariance before fitting.
Trees have bounded leaves; the residual cannot introduce unbounded far-distance
growth. A common null residual may be subtracted from every score in a group
without changing probabilities; no coordinate/node/time manipulation is allowed.

First use only the already frozen20-group fitting packet and original full CPU
smoke model. Pin all hashes. Require native/portable/JSON score and decision
replay, exact candidate coverage, original input preservation, actual training
loss reduction and label-free score replay. Persist each25-tree native state
and terminal models.300second CPU watchdog. No held-out quality claim from this
fit, no GPU/cloud mutation and no shared-environment changes. Dependencies live
in a new ignored local pair-tree-venv: Python3.12.10, NumPy2.5.3, SciPy1.18.1,
XGBoost3.4.1 (Apache-2.0). Windows wheel SHA256:
2d30fa513673101f542fdcbd18f30c8f96c064046f798635ac08663e9969f81b.

Only after this smoke and independent full linear verification succeed may a
first11-movie fitting-only resource profile be considered. It must use the
held-out-matched linear baseline/projection trained on exactly those11movies;
never fit a global baseline before cross-validation. No full12fold launch until
real profile timing/memory supports a declared cap. No hyperparameter retries
based on quality. The original five LOMO gates, complete-movie official scorer,
embryo and offline runtime gates remain mandatory for any later promotion.

References checked September10:
[custom-objective constraints and diagonal bounds](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html),
[parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[package provenance](https://pypi.org/project/xgboost/3.4.1/).
