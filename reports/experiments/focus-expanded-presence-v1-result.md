# Expanded-data presence experiments: rejected

The exact-replay summary/1 completed135.844s launcher,85.491s worker. All four
old fitting NPZs replayed exactly, followed by eight new fitting movies. Host
verified actual runtime/model/labels and reused identical old diagnostics.
Summary worker SHAee040d09a5fd9f5c08107da5b5ff29222817ce411da5819fbe1aebe7a55b2aa4.

Twelve fitting movies:10754knownparent/161knownabsent. Same four diagnostic
movies:2645knownparent/27knownabsent. Lower NLL is better; these are supervised
training-domain diagnostics, NOT complete-movie tracking or leaderboard scores.

| Method | Diagnostic NLL | Correct parent | Correct absent | Gate |
|---|---:|---:|---:|---|
| Original neural control |0.219009|2585|12|Reference|
| Physical control |0.333915|2500|18|Reference|
| Twelve-fit linear offset |0.186543|2602|7|FAIL|
| Twelve-fit fixed shallow trees |0.185738|2603|6|FAIL|

Both expanded models reduce NLL and improve parent predictions, but substantially
worsen missing-parent correctness. Neither is eligible for source extension or
submission. No acceptance gate or inference threshold was changed. No remaining
target movies opened. Do not extend these rejected models unchanged.

Linear CPU verification/fit elapsed67.735s; resultSHAad1a546a0c903358c37ab20aee206cd0c617b093a1aa122254721f178ca70dd0,
fitSHAb07f912a7eea8d81fabf8934f42724a4b0cc4a3a5cfc334bda3f398f27d1c272.
Tree CPU verification/fit elapsed32.391s, same original100trees/depth2/settings,
portable native-logit parity2.665e-15. ResultSHA4d0e314aa08e23233185fc8b5054b3d476f973a702bf0b4033c146ced723f6c8,
modelSHA7e4adeb894981ad8ab3d6677e1e7aeae8ba73f6e201dd5c9a5383695237acbfa.
Both complete fits were persisted before corrected diagnostic prediction.

Next intervention is association-head training with a fitting-count-derived
known-null loss weight8.1728227104, not another presence-only calibration.
CPU Torch smoke passed and actual downloaded runtime/results verified:
weight1 reproduces original loss exactly,unknown gradients exactly zero,
known-null gradient direction and division labels correct, synthetic loss
4.313976->0.351225. No real weighted training or GPU launch yet.
