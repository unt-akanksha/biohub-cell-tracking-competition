# trackastra-graph-finetune-v3

Status: staged independent association candidate; queued behind the active
Spotiflow detector run to enforce sequential GPU use.

V3 keeps the repaired offline numerical stack and complete clean validator from
V2, then adds a learned stage rather than more public-output post-processing:

- graph-only pretraining on 384 deterministic CC0 six-frame sequences;
- the audited one-time Y/X coordinate repair;
- a 0.063812 synthetic division-prior multiplier;
- 1,200 synthetic steps followed by 5,000 real Biohub graph steps; and
- initial, post-synthetic, post-real, and four-complete-movie validation.

The 18.5 GB synthetic output remains attached and only NPZ graph members are
read. The run uses one T4, Internet/TPU off, a 14,100-second hard stop, and no
competition submission code.
