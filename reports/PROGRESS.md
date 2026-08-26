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