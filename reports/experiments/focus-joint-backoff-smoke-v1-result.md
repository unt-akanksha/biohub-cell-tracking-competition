# Joint encoder/head smoke: numerical repair passed

Originalsmoke/1 failed safely before update2 on the1160x1187largest fitting
pair, after exact replay of all4 original images/features/logits and one finite
jointupdate. Actualfailedsource/log/terminal verified; launcher70.604seconds.
No larger run was launched from that failure.

The new backoffsmoke/1 completed29.859worker/82.027launcher seconds. Same
original model,4stresspairs,loss,learningrates,initialscale65536 and clipping.
Step2 overflow skipped optimizer with unchanged parameters, scale65536->32768;
step3 similarly32768->16384. Both then retried the same sample/RNG and completed
finiteupdates. Steps1/4needed no retry. Four successfulupdates, both encoder
andhead tensors changed; detectorhead andembeddedflow remained identical.
This confirms bounded scale recovery solved the observed probe failure. It is
not a guarantee that every later numerical failure is recoverable.

Exact original FP32image/features/logits replay passed on all4pairs. Final
checkpoint exactreal-image FP32reload passed. Peakallocated2,635,035,136bytes
(about2.45GiB),reserved3,542,089,728bytes on15,636,037,632byteT4. NoOOM or
node truncation. No diagnostic/source/target scoring or submission.

Host verifier checked actualruntime/stagedidentity, fullupstreamfeatureevidence,
exactstresspairselection, actualcheckpoint hash, two boundedscale-skips and
unchanged-parameter assertions. Guard remained enabled. ReceiptSHA
1e904d0d6bd34c18201dceee72034548fe3ab1316bd92562d821af23078f4d51;
workerresult392610c20a2b8e803e4913c1094e1e5fbf248778b836ca848c5b36502b518e6d.
Checkpoint57d60a70bc9276ff08748470ece13cb466fbd233d5798869d2975cc7b728ed23.

Eligible for a separately bounded jointtraining run, not a qualitypromotion.
Bothsmokejobs complete/terminal, checkpoints/logs saved. RSNA untouched.
