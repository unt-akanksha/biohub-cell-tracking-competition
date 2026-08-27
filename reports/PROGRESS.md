# Biohub Experiment Progress

Public leaderboard score is non-authoritative and cannot independently promote a run.

## Highest-Value Next Experiment

- Run: `prior-zebrahub-selective-ssm-medium`
- Hypothesis: ZebraHub selective_ssm_medium parent-ranking gains survive reciprocal competition calibration and exact complete-graph OOF scoring.
- Next gate: `reciprocal_competition_calibration_and_complete_graph_exact_metric`

## Run History

| Run | Parent | Hypothesis | State | Exact evidence | Public score (non-authoritative) | GPU hours actual / declared | Decision | Failure / failed gate | Next gate |
|---|---|---|---|---|---:|---:|---|---|---|
| `prior-dense-motion-state-guard` | — | A strict dense-motion topology state guard improves pooled exact OOF without division regression. | completed | {"division_delta":"0","pooled_score_delta":"0.000367"} | — | unknown / unknown | retain | — | — |
| `prior-pairwise-graph-policy` | prior-dense-motion-state-guard | A broader pairwise graph policy improves pooled score without damaging rare divisions. | completed | {"division_jaccard_delta":"-0.001364","pooled_score_delta":"0.001454"} | — | unknown / unknown | retire | — | — |
| `prior-frozen-topology-replication` | — | Frozen dense topology replication yields a policy that clears the bilateral exact gate. | rejected | not recorded | — | unknown / unknown | retire | No tested frozen dense topology policy passed. | — |
| `prior-vjepa-parent-ranking` | — | V-JEPA parent-ranking features improve both held-out embryos consistently. | completed | {"embryo_behavior":"asymmetric"} | — | unknown / unknown | inconclusive | — | — |
| `prior-sea-raft-parent-ranking` | — | SEA-RAFT parent-ranking features improve both held-out embryos consistently. | completed | {"embryo_behavior":"asymmetric"} | — | unknown / unknown | inconclusive | — | — |
| `prior-hoct-dense-movie` | — | HOCT can process the dense movie within Kaggle memory after fallback patching. | failed | not recorded | — | 2.0 / unknown | retire | Out of memory on a dense movie after fallback patching. | — |
| `prior-ranker-coverage-failure` | — | The sparse-PU ranker completes with invariant output movie coverage. | failed | not recorded | — | 11.3 / unknown | retire | Output movie coverage changed. | — |
| `prior-zebrahub-selective-ssm-medium` | — | ZebraHub selective_ssm_medium parent-ranking gains survive reciprocal competition calibration and exact complete-graph OOF scoring. | incomplete | {"mean_net_recoveries":"51.7","nearest_parent_top1":"0.932190","parent_top1":"0.932448"} | — | unknown / unknown | retain | — | reciprocal_competition_calibration_and_complete_graph_exact_metric |
| `phase1-live-guard-readonly-smoke` | — | Read-only Phase 1 live guard verification; no kernel launch is authorized or executed. | registered | not recorded | — | unknown / 0.01 | — | — | — |
| `public-0927-clean-repro-v1` | — | Intensity-weighted centroid refinement plus constrained division repair on the clean dual-seed harmonic TemporalUNet3D pipeline improves the owned public score beyond 0.913 without metric manipulation. | registered | not recorded | — | unknown / 1.00 | — | — | — |
| `public-0927-clean-repro-v2` | public-0927-clean-repro-v1 | The Windows-safe fork reproduces the clean 0.927 dual-seed harmonic TemporalUNet3D pipeline and improves the owned public score beyond 0.913 without metric manipulation. | completed | {"by_embryo":{"44b6":{"adjusted_edge_jaccard":"0.8910900942367352","division_jaccard":"0.0","proxy_score":"0.8910900942367352"},"6bba":{"adjusted_edge_jaccard":"0.9365550101410081","division_jaccard":"0.25","proxy_score":"0.9615550101410081"}},"by_fold":{"44b6_12dfb391":{"score":"0.934126868799285"},"44b6_267148e4":{"score":"0.7802295420408002"},"6bba_062c8d37":{"score":"1.0770587342596099"},"6bba_07e24132":{"score":"0.8356295217465801"}},"division_counts":{"fn":4,"fp":2,"tp":1},"evidence_scope":"four complete embryo-held training movies; exact owned output matches audited upstream output","note":"node_recall is null because the compact upstream validator did not emit the ground-truth node denominator; no value is imputed.","pooled":{"adjusted_edge_jaccard":"0.9151575278536992","division_jaccard":"0.14285714285714285","edge_jaccard":"0.9104539775093711","node_recall":null,"proxy_score":"0.9294432421394134"},"run_id":"public-0927-clean-repro-v2","schema_version":1,"validator_results_sha256":"b2b18eaeff608dae26202fe2a2c42aeac1987e0f592becd3dc75a2e0e490e777","worst_movie_delta":"-0.14921370009861323"} | — | 0.5616 / 1.00 | retain | — | — |
| `centroid-division-ablation-v1` | public-0927-clean-repro-v2 | Complete-movie validation aligned to the test-path centroid refinement will identify whether centroid refinement and safe-division repair improve the clean dual-seed harmonic pipeline without metric manipulation. | registered | not recorded | — | unknown / 1.00 | — | — | — |
| `trackastra-graph-finetune-v1` | public-0927-clean-repro-v2 | Biohub fine-tuning of the 27.5M-parameter four-frame Trackastra CTC association transformer, with detector-noise augmentation, physical anisotropy, dense tiling, hard negatives, and division upweighting, will improve clean complete-movie generalization over the public 0.927 pipeline's smaller two-frame linker. | failed | not recorded | — | 0.01 / 4.00 | — | Kaggle setup failed in 32 seconds because offline dependency resolution upgraded NumPy from the live kernel's 2.0.2 to 2.4.6, leaving SciPy loaded against an incompatible NumPy runtime. Independent input audit also found the cached fourth validator GEFF incomplete. Retry pins NumPy 2.0.2 and uses a fresh four-graph download that each loads end-to-end. | — |
| `trackastra-graph-finetune-v2` | trackastra-graph-finetune-v1 | With NumPy pinned to the live Kaggle SciPy-compatible runtime and all four validator graphs complete, Biohub fine-tuning of the 27.5M-parameter four-frame Trackastra association transformer will improve clean complete-movie generalization over the public pipeline's smaller two-frame linker. | failed | not recorded | — | 0.0015 / 4.00 | — | Offline dependency resolver rejected tracksdata numpy>2 against pinned live NumPy 2.0.2 before training; terminal evidence confirms zero model steps. | — |
| `spotiflow-detector-acceptance-v1` | public-0927-clean-repro-v2 | A 35.5M-parameter official Spotiflow 3D detector, selected on eight disjoint Biohub fields and calibrated only from image responses plus organizer node-count metadata, can match or improve annotated-node recall and density stability versus the public pipeline's 8.3M detector on four complete clean heldout movies. | registered | not recorded | — | unknown / 2.00 | — | — | — |
| `spotiflow-detector-acceptance-v2` | spotiflow-detector-acceptance-v1 | With the Kaggle numerical stack preserved, a 35.5M-parameter official Spotiflow 3D detector selected on eight disjoint Biohub fields and calibrated only from image responses plus organizer node-count metadata can match or improve annotated-node recall and density stability versus the public pipeline's 8.3M detector on four complete clean heldout movies. | failed | not recorded | — | 0.0184 / 2 | — | Offline numerical and GEFF setup passed; evaluator subprocess did not inherit the notebook-only biohub_tracking source path, so inference stopped before the first model frame. | — |
| `spotiflow-detector-acceptance-v3` | spotiflow-detector-acceptance-v2 | After propagating the verified support source into the evaluator subprocess, the official 35.5M Spotiflow 3D detector selected on eight disjoint Biohub fields and calibrated only from images plus organizer counts can match or improve annotated-node recall and count stability versus the memorized public comparator on four complete candidate-disjoint movies. | completed | {"acceptance":{"annotated_gt_nodes":2357,"annotated_recall_delta":"-0.1913449299957573","baseline_annotated_node_recall":"0.96902842596521","baseline_matched_gt_nodes":2284,"movies":4,"per_movie_recall_delta":{"44b6_12dfb391":"-0.18274111675126903","44b6_267148e4":"-0.4428571428571429","6bba_062c8d37":"-0.05698924731182797","6bba_07e24132":"-0.3621169916434541"},"spotiflow_annotated_node_recall":"0.7776834959694527","spotiflow_matched_gt_nodes":1833},"by_embryo":{"44b6":{"baseline_node_recall":"0.9588014981273408","node_recall":"0.7078651685393258","node_recall_delta":"-0.25093632958801504"},"6bba":{"baseline_node_recall":"0.9775019394879751","node_recall":"0.8355314197051978","node_recall_delta":"-0.1419705197827773"}},"by_fold":{"44b6_12dfb391":{"delta":"-0.18274111675126903","node_recall":"0.7868020304568528"},"44b6_267148e4":{"delta":"-0.4428571428571429","node_recall":"0.4857142857142857"},"6bba_062c8d37":{"delta":"-0.05698924731182797","node_recall":"0.9354838709677419"},"6bba_07e24132":{"delta":"-0.3621169916434541","node_recall":"0.5766016713091922"}},"decision":{"drop_in_detector":"retire","next_test":"Dense corrected synthetic fine-tuning followed by the same clean acceptance protocol.","reason":"Official pretrained Spotiflow is materially below the public detector on every acceptance movie.","transfer_initialization":"retain_smfish_3d"},"division_counts":{"fn":null,"fp":null,"tp":null},"evaluator_elapsed_seconds":"102.45089798","launcher_terminal_sha256":"9f97cc3b5554cf5b096cd0a78c0d5a2af5a95381fc46b2900e92db785bb3ee59","pooled":{"adjusted_edge_jaccard":null,"division_jaccard":null,"edge_jaccard":null,"node_recall":"0.7776834959694527"},"remaining_gpu_hours":"29.32","result_sha256":"248177dcf6075241b38ec35cedca371ed18ddd55bf6153b375baafdb7286afdc","run_id":"spotiflow-detector-acceptance-v3","schema_version":1,"selection":{"annotated_node_recall":"0.688839615668884","model_name":"smfish_3d","normalization":"spotiflow_auto"},"status":"completed","submission_created":false,"terminal_elapsed_seconds":"220.303","worst_movie_delta":"-0.4428571428571429"} | — | 0.0612 / 2 | retire | — | — |
| `trackastra-graph-finetune-v3` | trackastra-graph-finetune-v2 | Geometry-correct CC0 sequence-graph pretraining before real Biohub fine-tuning improves clean complete-movie association and division proxy over the public baseline. | registered | not recorded | — | unknown / 4.00 | — | — | — |
| `spotiflow-synthetic-finetune-v1` | spotiflow-detector-acceptance-v3 | Dense geometry-correct CC0 adaptation closes enough of the observed Biohub domain gap for the selected 35.5M Spotiflow warm start to become a viable detector candidate. | completed | {"base_weight_sha256":"1fdfd62c89a007094870782c27da163052ed72952a72d29f12e6730514d3ad6d","best_observed_synthetic_validation":{"epoch":2,"val_acc":"0.8202484250068665","val_f1":"0.892428994178772","val_loss":"2.3044614791870117"},"best_weight_sha256":"8b786a300d4446ce86c407a61b58075472b55a5dd929255dbbc0c23e375b7eac","budget_stop_requested":false,"by_embryo":{},"by_fold":{},"division_counts":{"fn":null,"fp":null,"tp":null},"epochs":4,"final_epoch":{"flow_loss":"0.18817086517810822","heatmap_loss":"0.9745568633079529","train_loss":"1.1627275943756104","val_acc":"0.8056281805038452","val_f1":"0.8833800554275513","val_loss":"2.9409303665161133"},"global_steps":3072,"launcher_elapsed_seconds":"384.078","parameter_count":35489892,"pooled":{"adjusted_edge_jaccard":null,"division_jaccard":null,"edge_jaccard":null,"node_recall":null},"promotion_state":"awaiting frozen real-movie detector acceptance","quota_after_hours":"29.21","result_sha256":"c06f57265507f853152a1ace742b87e8e44a2a6d080f0074ec3468ea8f931919","run_id":"spotiflow-synthetic-finetune-v1","schema_version":1,"status":"completed","terminal_sha256":"fe5b1779573fc9361ed4dee9a7ca7d51627cf2898aac051cf68629c38b27ff17","trainer_elapsed_seconds":"300.888","weights_changed":true,"worst_movie_delta":null} | — | 0.1067 / 2.00 | — | — | — |
| `trackastra-graph-finetune-v4` | trackastra-graph-finetune-v3 | A synthetic-pretrained then Biohub-finetuned 27.5M Trackastra association model can improve the same-node public detector graph on a frozen two-movie acceptance split, enabling a non-replica submission candidate. | registered | not recorded | — | unknown / 4.00 | — | — | — |
| `spotiflow-finetuned-acceptance-v1` | spotiflow-synthetic-finetune-v1 | Dense synthetic adaptation improves real-movie recall over the selected pretrained Spotiflow warm start and may close the gap to the public detector without label-based threshold tuning. | completed | {"acceptance":{"annotated_gt_nodes":2357,"baseline_annotated_node_recall":"0.96902842596521","baseline_matched_gt_nodes":2284,"candidate_annotated_node_recall":"0.09249045396690708","candidate_matched_gt_nodes":218,"delta_vs_baseline":"-0.876537971998303","delta_vs_pretrained_spotiflow":"-0.6851930420025456","movies":4,"pretrained_spotiflow_annotated_node_recall":"0.7776834959694527"},"by_embryo":{"44b6":{"baseline_node_recall":"0.9588014981273408","node_recall":"0.20411985018726592","node_recall_delta":"-0.7546816479400749"},"6bba":{"baseline_node_recall":"0.9775019394879751","node_recall":"0.0","node_recall_delta":"-0.9775019394879751"}},"by_fold":{"44b6_12dfb391":{"delta":"-0.6928934010152284","node_recall":"0.2766497461928934"},"44b6_267148e4":{"delta":"-0.9285714285714286","node_recall":"0.0"},"6bba_062c8d37":{"delta":"-0.9924731182795699","node_recall":"0.0"},"6bba_07e24132":{"delta":"-0.9387186629526463","node_recall":"0.0"}},"decision":{"drop_in_detector":"retire","next_test":"Keep the public detector frozen and improve associations with density-matched Trackastra training.","reason":"Synthetic-only fine-tuning catastrophically reduced clean real-movie recall and produced zero detections on three acceptance movies at the frozen calibration rule.","transfer_initialization":"retire_synthetic_checkpoint"},"division_counts":{"fn":null,"fp":null,"tp":null},"evaluator_elapsed_seconds":"79.805730905","launcher_terminal_sha256":"b4209e99b520a1f4bd591a80384fcdac6016406be679a8f5f3e18599020f48ad","pooled":{"adjusted_edge_jaccard":null,"division_jaccard":null,"edge_jaccard":null,"node_recall":"0.09249045396690708"},"remaining_gpu_hours":"29.16","result_sha256":"4a7fbed512f7c69769a84addd0398676a881568ed677e7299e12d530d9d8fff5","run_id":"spotiflow-finetuned-acceptance-v1","schema_version":1,"status":"completed","submission_created":false,"terminal_elapsed_seconds":"162.778","worst_movie_delta":"-0.9924731182795699"} | — | 0.0452 / 2.00 | retire | — | — |
| `trackastra-graph-finetune-v5` | trackastra-graph-finetune-v4 | Density-matched distractor training lets a 27.5M Biohub-finetuned Trackastra improve complete-movie associations over the same-node public graph, enabling an independent non-replica candidate. | failed | not recorded | — | 0.0322 / 4.00 | — | Kaggle CUDA pre-training validation failed because probability-space BCE was executed inside autocast; no optimizer step or model artifact was produced. | — |
| `trackastra-graph-finetune-v6` | trackastra-graph-finetune-v5 | CUDA-safe density-matched fine-tuning of the 27.5M Trackastra model can learn independent associations that improve the exact processed-topology heldout proxy over the frozen public graph. | completed | {"actual_gpu_hours":"0.5619","by_embryo":{},"by_fold":{},"complete_movie_validation_sha256":"186350e9209d5f284da77e8769ca2fca47d8afb954edc4852cf71dcdbc1f2926","decision":{"next_test":"Run trackastra-raw-confidence-acceptance-v1 once; retire immediately unless its disjoint processed-topology acceptance gate passes.","reason":"Raw-graph validation failed, but the experiment contract explicitly made it provisional because the submission candidate uses the audited processed final topology with transferred raw confidences.","state":"retain_only_for_predeclared_exact_processed_topology_gate"},"division_counts":{"fn":3,"fp":5,"tp":0},"launcher_elapsed_seconds":"2022.8","launcher_terminal_sha256":"f9acbb15ea0ce7caa16ab627659e06b8996eb1a316e5f8e39d1962940058bfbb","model_sha256":"255b3600a4926db6af36a7bddf6f1d84b711fca597602a4bdc511f5f4fd9166f","parameter_count":27456880,"pooled":{"adjusted_edge_jaccard":"0.7799176488499155","division_jaccard":"0.0","edge_jaccard":null,"node_recall":null},"provisional_raw_graph_validation":{"acceptance_delta_vs_base_raw":"-0.06579806240598918","association_acceptance_passed":false,"base_acceptance_proxy":"0.8457157112559047","base_worst_movie":"0.8431698208567855","selected_acceptance_proxy":"0.7799176488499155","selected_worst_movie":"0.7349797799364375","selection_delta_vs_base_raw":"-0.0029035865932345306"},"public_leaderboard_used_for_selection":false,"real_steps":7000,"remaining_gpu_hours":"28.55","run_id":"trackastra-graph-finetune-v6","schema_version":1,"status":"completed","submission_created":false,"synthetic_steps":1200,"trainer_elapsed_seconds":"1695.045306839","training_diagnostics":{"final_window_loss":"0.05353495265201976","final_window_positive_probability":"0.843684564034144","initial_window_loss":"0.00953277984800177","note":"The tiny positive-only window check degraded after real fine-tuning and did not contain negatives; the complete-movie gate is the relevant provisional evidence.","post_synthetic_window_loss":"0.004059034490118729"},"training_terminal_sha256":"c6c2843a2640b7ab55dadf9f19e2b6535ece50845e15b4cee8498734ca123d23","worst_movie_delta":"-0.108190040920348"} | — | 0.5619 / 4.00 | retain | — | — |
| `hoct-probe-finetune-v1` | trackastra-graph-finetune-v6 | A Biohub-supervised linear probe on frozen HOCT edge-centric features, combined with raw detector confidence, improves clean final-topology association over the public comparator and node-centric Trackastra. | registered | not recorded | — | unknown / 4.00 | — | — | — |
| `trackastra-raw-confidence-acceptance-v1` | trackastra-graph-finetune-v6 | The independently fine-tuned 27.5M Trackastra checkpoint plus transferred raw detector confidence improves the exact processed final-topology association proxy on disjoint clean acceptance movies without leaderboard feedback. | failed | not recorded | — | 0.22264 / 1.50 | — | Frozen topology materializer omitted the hash-pinned public preset cell and selected DeepCenter checkpoint_last.pt epoch 500; expected node-count guard failed before Trackastra inference, so no scientific acceptance result or submission was produced. | — |
| `trackastra-raw-confidence-acceptance-v2` | trackastra-graph-finetune-v6 | The independently fine-tuned 27.5M Trackastra checkpoint plus transferred raw detector confidence improves exact processed final-topology association on disjoint clean acceptance movies; v2 repairs only the pre-inference frozen-preset materialization failure. | failed | not recorded | — | 0.175548 / 1.50 | — | Frozen public topology materialization produced 21,843 nodes for 44b6_267148e4, a 0.34% drift from the earlier validator artifact's 21,768 count; model inference never started. This is comparator-integrity failure, not Trackastra model evidence; no submission was created. | — |
| `hoct-multibackbone-probe-v1` | trackastra-graph-finetune-v6 | Complementary HOCT general and CTC backbones with independently fitted Biohub probes and a robust worst-embryo selection rule improve exact processed-topology association over the frozen public comparator without leaderboard feedback. | rejected | not recorded | — | 1.1933794444 / 4.00 | retire | Training completed and candidate recall was 0.99994, but the frozen general_ctc_blend_w0.75 Biohub probe regressed the base by 0.05384 on selection and 0.004340 on untouched acceptance. Acceptance adjusted edge Jaccard was 0.81754 versus 0.82188; division false positives rose from 3 to 10. No submission was created. | — |
| `spatialdino-appearance-validation-v1` | hoct-multibackbone-probe-v1 | Frozen 3D microscopy foundation-model appearance features can correct a small set of ambiguous parent crossings while preserving the strong detector topology and every node degree, improving clean held-out association beyond the public comparator without leaderboard feedback. | failed | not recorded | — | 0.000012 / 2.00 | inconclusive | Kaggle did not materialize the attached HOCT kernel output, so hoct_output discovery failed in setup after 0.043 seconds. SpatialDINO inference and validation never started; no submission was created. | — |
| `spotiflow-pu-adaptation-v1` | spatialdino-appearance-validation-v1 | Positive-unlabeled adaptation of the official 35.5M Spotiflow detector using conservative two-seed teacher consensus, forced organizer positives, weak-strong consistency, and frozen-base flow preservation can match the public detector's clean recall without copying public predictions. | rejected | not recorded | — | 0.15005 / 2.00 | retire | Independent PU training completed all 1,024 steps across 187 non-validation movies, but clean selection recall was 0.779239 versus the 0.80 pooled gate and 0.486438 versus the 0.65 worst-movie gate. Acceptance remained unopened and no submission was created. | — |
| `spatialdino-pu-adaptation-v1` | spotiflow-pu-adaptation-v1 | An independent 29.5M-parameter microscopy-pretrained SpatialDINO/UNETR hybrid, trained on conservative positive-unlabeled consensus across every non-validation movie, will improve clean complete-movie detection recall and complement Spotiflow without copying public predictions. | rejected | not recorded | — | 0.18 / 2.00 | retire | Independent SpatialDINO PU training completed 768 steps and passed pooled selection recall at 0.885638, but the worst complete movie reached 0.611212 versus the immutable 0.65 gate. Acceptance remained unopened and no submission was created. | — |
| `spatialdino-appearance-validation-v2` | spatialdino-appearance-validation-v1 | The unchanged frozen SpatialDINO degree-preserving appearance correction can improve clean held-out association once the exact completed HOCT topology is materialized through a private hash-bound dataset instead of Kaggle kernel-output attachment. | rejected | not recorded | — | 0.02 / 2.00 | retire | Frozen SpatialDINO appearance evaluated all 18 preregistered degree-preserving configurations, but zero candidate pairs qualified, zero swaps were made, and selection delta versus the base remained 0. Acceptance stayed unopened and no submission was created. | — |
| `spatialdino-pu-selective-distillation-v2` | spatialdino-pu-adaptation-v1 | A minority disagreement-weighted soft probability loss on two-seed-supported voxels will improve dense-movie localization enough for the independent SpatialDINO/UNETR detector to clear the unchanged worst-movie recall gate without copying public predictions. | rejected | not recorded | — | 0.18 / 2.00 | retire | Selective distillation completed 768 paired steps and left pooled recall effectively unchanged at 0.885906, but regressed the limiting movie from 0.611212 to 0.602170 versus the immutable 0.65 gate. Acceptance remained unopened and no submission was created. | — |
| `lsm-fm-pu-adaptation-v1` | spatialdino-pu-selective-distillation-v2 | A geometry-matched 3D light-sheet foundation model will improve dim-cell localization beyond the retired 2D SpatialDINO hybrid while retaining conservative independent PU supervision. | failed | not recorded | — | 0.04 / 2.00 | — | Kaggle setup verified MONAI in the notebook, but the trainer subprocess omitted the verified MONAI import root from PYTHONPATH and failed before model construction or any optimizer step; no checkpoint, validation, or submission artifact was produced. | — |
| `lsm-fm-pu-adaptation-v2` | lsm-fm-pu-adaptation-v1 | After propagating the already hash-verified MONAI import root into trainer and evaluator subprocesses, the unchanged geometry-matched 3D LSM-FM experiment can produce its first scientific localization evidence. | rejected | not recorded | — | 0.21 / 2.00 | retain | The independent 3D LSM-FM detector completed 768 steps and improved the limiting movie to 0.622061 versus SpatialDINO v1 0.611212 and v2 0.602170, but remained below the immutable 0.65 worst-movie gate. Pooled recall was 0.884206, acceptance stayed sealed, and no submission was created. | — |
| `lsm-fm-image-text-pu-adaptation-v1` | lsm-fm-pu-adaptation-v2 | A larger feature-36 light-sheet image-text SwinUNETR with broader and deeper Biohub PU adaptation will lift the limiting movie above the unchanged 0.65 selection gate without copying public outputs. | rejected | not recorded | — | 0.57 / 2.00 | retain | Feature-36 completed all 3072 steps and improved pooled recall to 0.889128 and hard-movie recall to 0.641953, but remained 14 matches below the immutable 0.65 gate. Acceptance stayed sealed; no public predictions, leaderboard selection, or submission were used. | — |
| `lsm-fm-ensemble-validation-v1` | lsm-fm-image-text-pu-adaptation-v1 | A fixed equal-probability ensemble of independently pretrained LSM-FM feature-24 and feature-36 detectors plus a single global sub-voxel refinement can recover the 14 hard-movie matches needed to cross the unchanged 0.65 gate without retraining or leaderboard selection. | rejected | not recorded | — | 0.29 / 1.00 | retain | No predeclared candidate crossed the immutable 0.65 worst-movie gate; feature36 control remained best at 0.641953 and acceptance stayed sealed. | — |
| `lsm-fm-image-text-localization-refinement-v1` | lsm-fm-ensemble-validation-v1 | A single global raw-intensity-aware or wider-radius sub-voxel refinement of frozen feature-36 peaks will recover the 14 limiting-movie matches needed to cross 0.65 without material regression elsewhere. | rejected | not recorded | — | 0.30 / 1.00 | retain | Probability radius-2 reached 1078/1659=0.649789 on the limiting movie, exactly one match below the immutable 0.65 gate; acceptance remained sealed and no submission was created. | — |
| `lsm-fm-center-enhancement-v1` | lsm-fm-image-text-localization-refinement-v1 | A learned count-preserving 3D center-enhancement head trained only on one-to-one sparse annotation matches from non-validation movies will add the single limiting-movie match needed beyond probability_r2_p2 without a material per-movie regression. | rejected | not recorded | — | 0.44 / 1.25 | retain | The learned residual improved pooled selection matching by 32 nodes but regressed the limiting movie from 1078 to 1062 matches at full scale; half scale reached 1077. Neither crossed the immutable 0.65 gate, acceptance stayed sealed, and no submission was created. | — |
| `lsm-fm-public-node-refinement-v1` | public-0927-clean-repro-v2 | A single global coordinate blend from the frozen high-recall public graph toward the independently trained 35.1M-parameter LSM-FM feature-36 probability centroid will strictly increase matched held-out nodes while preserving every node ID, edge, frame assignment, and node count. | failed | not recorded | — | 0.04 / 1.00 | — | Kaggle run aborted after 105.603 seconds before scientific evaluation because the attached graph runtime contained only four acceptance control GEFFs; the first selection control 44b6_d29c9ab2.geff was absent. No result, acceptance labels, or submission artifact was created. | — |
| `lsm-fm-public-node-refinement-v2` | lsm-fm-public-node-refinement-v1 | A conservative radius-2, power-2, 0.25 blend from frozen public nodes toward an independently trained 35.1M-parameter LSM-FM probability centroid will improve integer-submission-space organizer score without changing node IDs, counts, frame assignments, or graph topology. | rejected | not recorded | — | 0.07 / 1.00 | retire | The predeclared LSM-FM coordinate blend gained four pooled matches and improved mean localization distance, but lost one match on 6bba_07e24132 (-0.0027855 recall), exceeding the frozen -0.002 per-movie regression limit. No exact scoring, production materialization, or submission was performed. | — |
| `trackastra-dual-fold-synthetic-v1` | trackastra-graph-finetune-v6 | Correct native synthetic geometry plus long 95/5 synthetic-real replay and frame-global motion augmentation improves association and division ranking without v6 real-only drift. | running | not recorded | — | unknown / 4.00 | — | — | — |
| `temporal-patch-dual-fold-v1` | trackastra-dual-fold-synthetic-v1 | A project-authored physical-scale temporal 3D appearance representation resolves ambiguous geometrically plausible parent-child links beyond the coherent Trackastra geometry control without public predictions or leaderboard selection. | running | not recorded | — | unknown / 11.00 | — | — | — |
| `temporal-patch-dual-fold-blend-v1` | temporal-patch-dual-fold-v1 | A fixed, cleanly held-out blend of project-authored temporal appearance evidence with coherent Trackastra geometry improves both reciprocal embryo folds without per-movie regression. | registered | not recorded | — | unknown / 6.00 | — | — | — |

## Exact Evidence Details

### prior-dense-motion-state-guard

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `{"division_delta":"0","pooled_score_delta":"0.000367"}`
- Decision evidence: `["exact_oof:pooled_delta_0.000367","division:no_regression"]`

### prior-pairwise-graph-policy

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `{"division_jaccard_delta":"-0.001364","pooled_score_delta":"0.001454"}`
- Decision evidence: `["exact_oof:pooled_delta_0.001454","division:delta_-0.001364"]`

### prior-frozen-topology-replication

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `not recorded`
- Decision evidence: `["audit:no_passing_policy"]`

### prior-vjepa-parent-ranking

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `{"embryo_behavior":"asymmetric"}`
- Decision evidence: `["embryo_audit:asymmetric"]`

### prior-sea-raft-parent-ranking

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `{"embryo_behavior":"asymmetric"}`
- Decision evidence: `["embryo_audit:asymmetric"]`

### prior-hoct-dense-movie

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `not recorded`
- Decision evidence: `["failure:oom_dense_movie"]`

### prior-ranker-coverage-failure

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `not recorded`
- Decision evidence: `["failure:output_movie_coverage_changed"]`

### prior-zebrahub-selective-ssm-medium

- Authorized for submission: `false`
- Imported audit: `true`
- Evidence: `{"mean_net_recoveries":"51.7","nearest_parent_top1":"0.932190","parent_top1":"0.932448"}`
- Decision evidence: `["external_holdout:all_maturation_gates_passed_three_seeds","authorization:false"]`

### phase1-live-guard-readonly-smoke

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### public-0927-clean-repro-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### public-0927-clean-repro-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `{"by_embryo":{"44b6":{"adjusted_edge_jaccard":"0.8910900942367352","division_jaccard":"0.0","proxy_score":"0.8910900942367352"},"6bba":{"adjusted_edge_jaccard":"0.9365550101410081","division_jaccard":"0.25","proxy_score":"0.9615550101410081"}},"by_fold":{"44b6_12dfb391":{"score":"0.934126868799285"},"44b6_267148e4":{"score":"0.7802295420408002"},"6bba_062c8d37":{"score":"1.0770587342596099"},"6bba_07e24132":{"score":"0.8356295217465801"}},"division_counts":{"fn":4,"fp":2,"tp":1},"evidence_scope":"four complete embryo-held training movies; exact owned output matches audited upstream output","note":"node_recall is null because the compact upstream validator did not emit the ground-truth node denominator; no value is imputed.","pooled":{"adjusted_edge_jaccard":"0.9151575278536992","division_jaccard":"0.14285714285714285","edge_jaccard":"0.9104539775093711","node_recall":null,"proxy_score":"0.9294432421394134"},"run_id":"public-0927-clean-repro-v2","schema_version":1,"validator_results_sha256":"b2b18eaeff608dae26202fe2a2c42aeac1987e0f592becd3dc75a2e0e490e777","worst_movie_delta":"-0.14921370009861323"}`
- Decision evidence: `["reports/experiments/public-0927-clean-repro-v2-output-audit.json","exact_owned_submission_hash_matches_audited_public_candidate","complete_four_movie_heldout_validator_proxy_0.929443","kaggle_submission_55784044_pending"]`

### centroid-division-ablation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### trackastra-graph-finetune-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### trackastra-graph-finetune-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### spotiflow-detector-acceptance-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### spotiflow-detector-acceptance-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### spotiflow-detector-acceptance-v3

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `{"acceptance":{"annotated_gt_nodes":2357,"annotated_recall_delta":"-0.1913449299957573","baseline_annotated_node_recall":"0.96902842596521","baseline_matched_gt_nodes":2284,"movies":4,"per_movie_recall_delta":{"44b6_12dfb391":"-0.18274111675126903","44b6_267148e4":"-0.4428571428571429","6bba_062c8d37":"-0.05698924731182797","6bba_07e24132":"-0.3621169916434541"},"spotiflow_annotated_node_recall":"0.7776834959694527","spotiflow_matched_gt_nodes":1833},"by_embryo":{"44b6":{"baseline_node_recall":"0.9588014981273408","node_recall":"0.7078651685393258","node_recall_delta":"-0.25093632958801504"},"6bba":{"baseline_node_recall":"0.9775019394879751","node_recall":"0.8355314197051978","node_recall_delta":"-0.1419705197827773"}},"by_fold":{"44b6_12dfb391":{"delta":"-0.18274111675126903","node_recall":"0.7868020304568528"},"44b6_267148e4":{"delta":"-0.4428571428571429","node_recall":"0.4857142857142857"},"6bba_062c8d37":{"delta":"-0.05698924731182797","node_recall":"0.9354838709677419"},"6bba_07e24132":{"delta":"-0.3621169916434541","node_recall":"0.5766016713091922"}},"decision":{"drop_in_detector":"retire","next_test":"Dense corrected synthetic fine-tuning followed by the same clean acceptance protocol.","reason":"Official pretrained Spotiflow is materially below the public detector on every acceptance movie.","transfer_initialization":"retain_smfish_3d"},"division_counts":{"fn":null,"fp":null,"tp":null},"evaluator_elapsed_seconds":"102.45089798","launcher_terminal_sha256":"9f97cc3b5554cf5b096cd0a78c0d5a2af5a95381fc46b2900e92db785bb3ee59","pooled":{"adjusted_edge_jaccard":null,"division_jaccard":null,"edge_jaccard":null,"node_recall":"0.7776834959694527"},"remaining_gpu_hours":"29.32","result_sha256":"248177dcf6075241b38ec35cedca371ed18ddd55bf6153b375baafdb7286afdc","run_id":"spotiflow-detector-acceptance-v3","schema_version":1,"selection":{"annotated_node_recall":"0.688839615668884","model_name":"smfish_3d","normalization":"spotiflow_auto"},"status":"completed","submission_created":false,"terminal_elapsed_seconds":"220.303","worst_movie_delta":"-0.4428571428571429"}`
- Decision evidence: `["reports/experiments/spotiflow-detector-acceptance-v3-result.json"]`

### trackastra-graph-finetune-v3

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### spotiflow-synthetic-finetune-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `{"base_weight_sha256":"1fdfd62c89a007094870782c27da163052ed72952a72d29f12e6730514d3ad6d","best_observed_synthetic_validation":{"epoch":2,"val_acc":"0.8202484250068665","val_f1":"0.892428994178772","val_loss":"2.3044614791870117"},"best_weight_sha256":"8b786a300d4446ce86c407a61b58075472b55a5dd929255dbbc0c23e375b7eac","budget_stop_requested":false,"by_embryo":{},"by_fold":{},"division_counts":{"fn":null,"fp":null,"tp":null},"epochs":4,"final_epoch":{"flow_loss":"0.18817086517810822","heatmap_loss":"0.9745568633079529","train_loss":"1.1627275943756104","val_acc":"0.8056281805038452","val_f1":"0.8833800554275513","val_loss":"2.9409303665161133"},"global_steps":3072,"launcher_elapsed_seconds":"384.078","parameter_count":35489892,"pooled":{"adjusted_edge_jaccard":null,"division_jaccard":null,"edge_jaccard":null,"node_recall":null},"promotion_state":"awaiting frozen real-movie detector acceptance","quota_after_hours":"29.21","result_sha256":"c06f57265507f853152a1ace742b87e8e44a2a6d080f0074ec3468ea8f931919","run_id":"spotiflow-synthetic-finetune-v1","schema_version":1,"status":"completed","terminal_sha256":"fe5b1779573fc9361ed4dee9a7ca7d51627cf2898aac051cf68629c38b27ff17","trainer_elapsed_seconds":"300.888","weights_changed":true,"worst_movie_delta":null}`
- Decision evidence: `not recorded`

### trackastra-graph-finetune-v4

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### spotiflow-finetuned-acceptance-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `{"acceptance":{"annotated_gt_nodes":2357,"baseline_annotated_node_recall":"0.96902842596521","baseline_matched_gt_nodes":2284,"candidate_annotated_node_recall":"0.09249045396690708","candidate_matched_gt_nodes":218,"delta_vs_baseline":"-0.876537971998303","delta_vs_pretrained_spotiflow":"-0.6851930420025456","movies":4,"pretrained_spotiflow_annotated_node_recall":"0.7776834959694527"},"by_embryo":{"44b6":{"baseline_node_recall":"0.9588014981273408","node_recall":"0.20411985018726592","node_recall_delta":"-0.7546816479400749"},"6bba":{"baseline_node_recall":"0.9775019394879751","node_recall":"0.0","node_recall_delta":"-0.9775019394879751"}},"by_fold":{"44b6_12dfb391":{"delta":"-0.6928934010152284","node_recall":"0.2766497461928934"},"44b6_267148e4":{"delta":"-0.9285714285714286","node_recall":"0.0"},"6bba_062c8d37":{"delta":"-0.9924731182795699","node_recall":"0.0"},"6bba_07e24132":{"delta":"-0.9387186629526463","node_recall":"0.0"}},"decision":{"drop_in_detector":"retire","next_test":"Keep the public detector frozen and improve associations with density-matched Trackastra training.","reason":"Synthetic-only fine-tuning catastrophically reduced clean real-movie recall and produced zero detections on three acceptance movies at the frozen calibration rule.","transfer_initialization":"retire_synthetic_checkpoint"},"division_counts":{"fn":null,"fp":null,"tp":null},"evaluator_elapsed_seconds":"79.805730905","launcher_terminal_sha256":"b4209e99b520a1f4bd591a80384fcdac6016406be679a8f5f3e18599020f48ad","pooled":{"adjusted_edge_jaccard":null,"division_jaccard":null,"edge_jaccard":null,"node_recall":"0.09249045396690708"},"remaining_gpu_hours":"29.16","result_sha256":"4a7fbed512f7c69769a84addd0398676a881568ed677e7299e12d530d9d8fff5","run_id":"spotiflow-finetuned-acceptance-v1","schema_version":1,"status":"completed","submission_created":false,"terminal_elapsed_seconds":"162.778","worst_movie_delta":"-0.9924731182795699"}`
- Decision evidence: `["reports/experiments/spotiflow-finetuned-acceptance-v1-result.json",".biohub/cache/kernel-outputs/spotiflow-finetuned-acceptance-v1/spotiflow_finetuned_acceptance/finetuned_detector_acceptance.json"]`

### trackastra-graph-finetune-v5

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### trackastra-graph-finetune-v6

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `{"actual_gpu_hours":"0.5619","by_embryo":{},"by_fold":{},"complete_movie_validation_sha256":"186350e9209d5f284da77e8769ca2fca47d8afb954edc4852cf71dcdbc1f2926","decision":{"next_test":"Run trackastra-raw-confidence-acceptance-v1 once; retire immediately unless its disjoint processed-topology acceptance gate passes.","reason":"Raw-graph validation failed, but the experiment contract explicitly made it provisional because the submission candidate uses the audited processed final topology with transferred raw confidences.","state":"retain_only_for_predeclared_exact_processed_topology_gate"},"division_counts":{"fn":3,"fp":5,"tp":0},"launcher_elapsed_seconds":"2022.8","launcher_terminal_sha256":"f9acbb15ea0ce7caa16ab627659e06b8996eb1a316e5f8e39d1962940058bfbb","model_sha256":"255b3600a4926db6af36a7bddf6f1d84b711fca597602a4bdc511f5f4fd9166f","parameter_count":27456880,"pooled":{"adjusted_edge_jaccard":"0.7799176488499155","division_jaccard":"0.0","edge_jaccard":null,"node_recall":null},"provisional_raw_graph_validation":{"acceptance_delta_vs_base_raw":"-0.06579806240598918","association_acceptance_passed":false,"base_acceptance_proxy":"0.8457157112559047","base_worst_movie":"0.8431698208567855","selected_acceptance_proxy":"0.7799176488499155","selected_worst_movie":"0.7349797799364375","selection_delta_vs_base_raw":"-0.0029035865932345306"},"public_leaderboard_used_for_selection":false,"real_steps":7000,"remaining_gpu_hours":"28.55","run_id":"trackastra-graph-finetune-v6","schema_version":1,"status":"completed","submission_created":false,"synthetic_steps":1200,"trainer_elapsed_seconds":"1695.045306839","training_diagnostics":{"final_window_loss":"0.05353495265201976","final_window_positive_probability":"0.843684564034144","initial_window_loss":"0.00953277984800177","note":"The tiny positive-only window check degraded after real fine-tuning and did not contain negatives; the complete-movie gate is the relevant provisional evidence.","post_synthetic_window_loss":"0.004059034490118729"},"training_terminal_sha256":"c6c2843a2640b7ab55dadf9f19e2b6535ece50845e15b4cee8498734ca123d23","worst_movie_delta":"-0.108190040920348"}`
- Decision evidence: `["reports/experiments/trackastra-graph-finetune-v6-result.json"]`

### hoct-probe-finetune-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### trackastra-raw-confidence-acceptance-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### trackastra-raw-confidence-acceptance-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### hoct-multibackbone-probe-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/hoct-multibackbone-probe-v1-result.json",".biohub/cache/kernel-outputs/hoct-multibackbone-probe-v1/hoct_multibackbone_v1/complete_movie_validation.json"]`

### spatialdino-appearance-validation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `[".biohub/cache/kernel-outputs/spatialdino-appearance-validation-v1/launcher_terminal.json",".biohub/cache/kernel-outputs/spatialdino-appearance-validation-v1/biohub-spatialdino-appearance-validation-v1.log"]`

### spotiflow-pu-adaptation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/spotiflow-pu-adaptation-v1-result.json",".biohub/cache/kernel-outputs/spotiflow-pu-adaptation-v1/spotiflow_pu_validation/pu_detector_validation.json"]`

### spatialdino-pu-adaptation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/spatialdino-pu-adaptation-v1-result.json",".biohub/cache/kernel-outputs/spatialdino-pu-adaptation-v1/spatialdino_pu_validation/spatialdino_pu_validation.json"]`

### spatialdino-appearance-validation-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/spatialdino-appearance-validation-v2-result.json",".biohub/cache/kernel-outputs/spatialdino-appearance-validation-v2/spatialdino_appearance_v2/complete_movie_validation.json"]`

### spatialdino-pu-selective-distillation-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/spatialdino-pu-selective-distillation-v2-result.json",".biohub/cache/kernel-outputs/spatialdino-pu-selective-distillation-v2/spatialdino_pu_validation/spatialdino_pu_validation.json"]`

### lsm-fm-pu-adaptation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### lsm-fm-pu-adaptation-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/lsm-fm-pu-adaptation-v2-result.json"]`

### lsm-fm-image-text-pu-adaptation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/lsm-fm-image-text-pu-adaptation-v1-result.json"]`

### lsm-fm-ensemble-validation-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/lsm-fm-ensemble-validation-v1-result.json"]`

### lsm-fm-image-text-localization-refinement-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/lsm-fm-image-text-localization-refinement-v1-result.json"]`

### lsm-fm-center-enhancement-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/lsm-fm-center-enhancement-v1-result.json"]`

### lsm-fm-public-node-refinement-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### lsm-fm-public-node-refinement-v2

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `["reports/experiments/lsm-fm-public-node-refinement-v2-result.json"]`

### trackastra-dual-fold-synthetic-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### temporal-patch-dual-fold-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`

### temporal-patch-dual-fold-blend-v1

- Authorized for submission: `false`
- Imported audit: `false`
- Evidence: `not recorded`
- Decision evidence: `not recorded`


## CPU Acceptance Controls

- `phase2-cpu-control-20260824T094359Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260824T095539Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260824T101305Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260824T104104Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260824T150753Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T195412Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T202351Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T204326Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T205428Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T211832Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T213542Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T215425Z`: `failed`, accelerator `none`, promotion eligible `false`
- `phase2-cpu-control-20260825T224606Z`: `completed`, accelerator `none`, promotion eligible `false`

## Aggregate Exact Evaluations

- `phase2-control-evaluation-20260825T224606Z`: `completed`, decision `not recorded`