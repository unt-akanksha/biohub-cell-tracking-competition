# Temporal contextual final autochain v1

Status: exact-scoring/final-launch and verified-submission watchers are armed;
no final kernel run or competition submission exists at this record point.

After the two-GPU processed benchmark completes, the first watcher downloads
its actual Kaggle layout and runs the pinned public organizer scorer in the
locked evaluation environment against the four predeclared complete movies.
The exact gate requires a reproduced control score, improved pooled score, no
movie regression below `-0.002`, identical node recall, and changed edge sets.
It performs no parameter selection and reads no leaderboard result. Rejection
stops the chain.

Accepted evidence is staged and uploaded as the private
`indarkarhana/biohub-temporal-contextual-exact-acceptance-v3` dataset. The final
preflight binds that evidence to both transferred checkpoint hashes and the
Kaggle processed artifact. The watcher then waits for the full 12-hour notebook
ceiling with `0.0` hours reserved, validates the frozen two-T4 notebook and
metadata, and pushes exactly one final kernel version.

The second watcher polls only after that local launch terminal exists. A
completed final output is independently downloaded and checked for exact remote
kernel topology, launcher/report/candidate hash agreement, whole-movie
coverage, node preservation, changed edges, non-replica provenance, the unique
prospective submission authorization, and the daily submission limit. It then
submits that exact kernel version without a GPU-quota check and writes a receipt.

- Exact/final watcher SHA-256:
  `e4f00187f53ffb6a17634b52065d119bc181030c6bdbb3fe233e01eeb069fba6`
- Verified-submission watcher SHA-256:
  `7e08c8c05da318a72587e7ea5d7d5cf185fc2d1f9875b3c7b57dbe6f781f0e53`
- Exact/final watcher test SHA-256:
  `84e1d7a87d0465afff609794330fff2324bfb991e70749bfc3147bf01838cb09`
- Verified-submission watcher test SHA-256:
  `1ec906d82fd372378db72dfbb5883328694cae100f3e88068b25de1e2970b977`
- Focused tests: `19 passed`
- Final notebook SHA-256:
  `5dad76f56003be6f84e381dbf1a735cb2e091dcedf8f238edb184fd0091c03af`
- Final metadata SHA-256:
  `429ddae35e0633069a5ac44151cd6a1e4782ccd130b1f315d448eb43be8276fe`
- Exact/final watcher PID at arm time: `47508`
- Submission watcher PID at arm time: `47496`
- Final-launch terminal:
  `.biohub/automation/temporal-contextual-final-launch.json`
- Exact acceptance evidence:
  `.biohub/automation/temporal-contextual-exact-acceptance.json`
- Submission receipt:
  `.biohub/automation/temporal-contextual-submission-receipt.json`
