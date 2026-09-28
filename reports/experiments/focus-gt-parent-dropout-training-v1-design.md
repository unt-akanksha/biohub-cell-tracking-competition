# Fixed800step exact-GT candidate-dropout training

Requires actual successful exact-GT four-step smoke and host verification before
staging. Start original owned model, not a selected smoke/adaptation checkpoint.
Use the verified1121 original and1121 exact-GT augmented fitting pairs only.
Alternate odd original/even augmented updates with two independent shuffled
queues consuming Python RNG seed244691. No hard-example mining, loss selection
or changes driven by diagnostic outcomes in this fixed run.

Retain smoke-tested FP32head-only AdamW1e-4,weightdecay1e-4,clip1; encoder,
detector and flow frozen. Combined counts20387parent/1443null determine the
square-root class weight. Physical prior, null score and original unweighted
diagnostic code unchanged. Require initial complete control replay before
optimizer. Final800step model only: NLL below both controls, correctparent>=2585
and correctabsent>=18 on the same2645parent/27null real labels. No gate relaxation.

Save model/optimizer/RNG/both queues at4and every100steps, with strict real-packet
reload every time. Full records stay in result.json; console prints compact
progress and final diagnostic summary. No source/target expansion or submission
unless separately justified after a pass and complete-movie tracking evaluation.

Same private offline two-T4 environment,900second declared cap,780second worker
deadline/840second watchdog. Freshquota before launch must preserve8h after
0.25h worstcase. Sequential Biohub experiments; no shared cloud or RSNA mutation.
