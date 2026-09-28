# Pair appearance and LDA: verified execution, quality comparison running

September10 around19:26UTC. The preceding paper-answer turn restated already
available evidence and made no new goal progress. This continuation completed
GPU equivalence, exact CPU performance improvements, a live rules/discussion
audit, and launched the actual frozen24-fit comparison. Goal remains unfinished.

19:31UTC update: LDA12folds completed421s and host-verified145.125s.
NLL0.342701,parent9880/10754,absent84/161. All12movie NLLwins; absent115gate
fails, no promotion. Full72D arm still live in session2266, firstfold100iterations
at121.125fit seconds. See`focus-pair-appearance-lomo-v1-lda-result.md` for the
complete result, comparisons and hashes. Earlier partial counts below are
historical. Verifier39992 is terminal; do not restart the live training handle.

## Complete data and small tests

Twelve original fitting movies,10,915 known-target groups and4,691,320 full
candidate/null choices were preserved.64D pair descriptors use normalized
32D source/target channel differences/products, with exact zero null rows.
All descriptors and class moments were independently reconstructed from the
original cached packets; original eight features/labels replayed. Descriptor
storage1,200,977,920bytes; extraction77.578s; extraction+smokes95.047CPU seconds.

The full72D and LDA9D small fits used only the preselected20-target fitting
packet. Both passed analytic gradients, serialization and exact CPU replay.
Those fitting decisions are functionality checks, not generalization evidence.
Data receipt4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f;
verification823d8f64a4615ada45a220036a7125551ea3a997438d46b14d864bdaa988cab9.

## Actual bounded GPU smoke

Private offline`indarkarhana/biohub-focus-pair-appearance-gpu-smoke-v1/1`
launched after preceding Biohub job was confirmed COMPLETE and fresh quota was
8.37h. CLI runtime cap300s; worker240s less setup; emergency285s. No submission.
Returned state COMPLETE; actual launcher9.04094s, worker6.05740s. These are run
timings, not a claim that billed quota equals9seconds. Torch2.10.0+cu128,
twoT4 devices available, one used, peak allocated23,200,256bytes.

GPU optimizer times: full0.42698s, LDA0.11018s. At zero and stored CPU weights,
loss errors0, maximum gradient error2.99e-13, maximum logit error1.82e-12.
All20 optimized target decisions match CPU; fixed synthetic variable-size and
null-only groups also pass. GPU models/runtime/input packet were downloaded;
host replay verified exact data hashes, controls, choices and model objectives.
Host verification6996216cf467c4faa9ad64bc43caabafc3047f5c1bccf8eea36a93eaa73b9f2.
No full-size GPU quality result, new feature extraction or backbone training.

## CPU implementation speedup and real comparison

Original first-fold objective7.7-10.2s repeated projection/normalization every
optimizer call. Preparing transformed files once exactly reproduces the CPU
smoke's objective, gradient and optimized coefficient vectors. Persistent
readonly maps for full features and <=512MiB resident arrays for LDA reduce
measured objective calls to full0.672-0.906s, LDA0.328-0.453s. Every complete
first-fold transform and original zero objective/gradient norm replays.
No scientific method or acceptance threshold changed.

Prepared profile8a26587a4c40e44c15db2457554e46e2486ef6301f18d0261ba1cb1369a35cce;
resident profilee9d65e8e07f867dfd4de4cc5698daf03a84f0317a874937ec6d86101ac2f3b6d.
Forty-six targeted tests pass19.36s before launch. Verifier syntax and tracked
diff checks pass. Existing unrelated dirty worktree changes preserved.

Sequential CPU run`focus-pair-appearance-lomo-v1` is live in session2266:
LDA12folds first, then full12folds; two numeric threads,9000s hard cap. Each
fold-only scaler/projection/weight/head is saved before its held-out evaluation.
Each fold result is persisted immediately. Estimate1-1.5hours, not a guarantee.
No final model export until unchanged gates and host verification pass.
Models/results/cache: `.biohub/cache/focus-pair-appearance-lomo-v1`.

At the latest read, eight LDA folds were complete and the ninth was fitting.
Do not promote from partial aggregates or interpret correction-held-out movies
as encoder-held-out embryos. Full arm will run regardless of LDA quality.

## Shared resources, rules and sources

19:17UTC read-only EC2 query confirms Antelume stopped,g5.xlarge,no public IP.
No instance, RSNA, shared environment or other project mutation. Kaggle8.22h
at that check; account usage continues outside this completed smoke. No further
Biohub GPU launch is needed for the CPU comparison; recheck before any future
GPU use and preserve the8h reserve.

Authenticated rules/all8page refresh completed19:23UTC; rules and Code
Requirements exactly unchanged. Full text of3 relevant discussion threads read;
no notebook/configuration or participant score adopted. Newer revision of an
already excluded exploit notebook skipped before pull. See
`biohub-rules-refresh-20260910-1923.md` for sources and caveats.

No qualified new candidate or submission:0/5 today. The strongest clean final
candidate, full metric/embryo validation and offline submission remain unproven.
