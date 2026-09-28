# Experimental birth/death dominance: not deployed

The frozen solver chooses one event per parent and one explanation per child.
A continuation can be replaced by parent-death plus child-birth; a fork by
parent-death plus two child-births. These replacements cover the same equality
constraint rows. When all replacement options are allowed and their combined
score is strictly better (existing relative1e-10tie tolerance), the original
event belongs to no optimal assignment. This is exact inference pruning, not
a learned threshold, biological birth relabeling or metric-dependent rule.

The new helper first runs the existing vectorized fork pruner, then prunes
these additional dominated options. Forbidden birth/death replacements cannot
be used. Six tests pass including exhaustive feasible assignments for30random
small problems: ALLoriginal optima survive, not merely one objective value.

Real small-case smoke passed both anchor and final-source44weights. Full
benchmark61489 TERMINAL success,10.078seconds total,3fixed cases x2models.
All6exact assignments and objective values match. Pruning+solve time5.1051s ->
3.1127s,1.64x in this component benchmark only. Largest case91,317options:
anchor active13,905 ->2,446; learned13,912 ->6,086. Case construction and feature
extraction are excluded from those component timings; no end-to-end speed claim.

Complete small-movie portable replay89395 TERMINAL success. Its exact99transition
graph and decision/score records match after excluding intentionally changed
pruning inventory fields. Additional6,333options removed. Inference4.937s vs
earlier3.922s reference: slower in this small shared-CPU timing, so no speedup
is claimed from it. No node, coordinate, protected incidence or fallback change.

The subsequent two-complete-movie replay uses a PRIVATE cloned inference
namespace; original modules, live training/evaluator, the drafted worker and
pending submission remain unchanged. It uses frozen final44weights and the
previously fixed small/dense movies, never GT. Full result remains pending.

## 21:52UTC: two complete movies replay exactly

Full replay51662 is TERMINAL success,241.609seconds including inputs and feature
generation. Small6bba_55b7eebe inference3.469s,6,333extra options removed;
dense6bba_57b7cc1e211.234s,669,938extra options removed. Both complete graphs and
decision/gain records match the frozen portable reference,99transitions each,
no fallback or cap. Only pruning inventory fields intentionally differ.

The dense previous reference took377.329s; this run took211.234s (~44%less),
but these were separate shared-CPU observations, not a controlled end-to-end
Kaggle speed claim. The paired six-case pruning+solve benchmark is the tighter
local speed measurement. Full receiptSHA
`756222160dc4d8e825c0e68292c5b8e8da58523550ef319eca63a423e8a34b3b`;
case benchmarkSHA`cf6ec62dbf3937fb53705362030e0b48cf42a3479f443b88bc22932539f6f061`.
Fourteen combined worker/pruning/portable tests passed in3.33seconds.

Next is a paired, cache-only public-test timing test. It uses our existing
network/structured outputs from the verified public run; no image-network GPU
rerun, labels, scoring, staging mutation, or submission. Both original-vectorized
and extra-pruned learned heads run with the same10/600limits and fixed44weights.
Every public graph pair must match; partial/fallback output is not acceptance.
Arm order alternates in the four-movie run to reduce systematic timing bias.

## 22:15-22:24 UTC: paired public benchmark and standalone replay complete

Paired run8284 TERMINAL success,1058.438s. Four complete graphs exact,
99transitions each, no solver fallback/stage exhaustion, no GT opened.
Event-inference seconds (reference ->null-pruned):

- 6bba_05b6850b:6.328 ->6.421 (slightly slower).
- 44b6_0b24845f:39.718 ->34.406.
- 44b6_0113de3b:92.156 ->75.875.
- 6bba_05db0fb1:408.891 ->338.687.

Sum547.093 ->455.389seconds,16.762%less event-inference time; shared feature
construction40.298s excluded. Shared localCPU observations, not a Kaggle
end-to-end forecast. Paired reportSHA
`a0dca4c1993f709482f2d98fc3462e0cf159cf27dccb2cffc4042306fb0fcd2e`.

New builder scripts/build-trajectory-event-null-portable-v2.py packages the
exact pruner as standalone stdlib/NumPy/SciPy inference. Only the old
allowed_options function is renamed fork_pruning and the tested wrapper added;
all other function ASTs match. No project imports, model fitting or GT in code.
14combined pruning/standalone/worker tests passed19.74s. BuilderSHA
`2ec6470444db92903ac2a17387695c3fefcbe640fcd7351022b0ccaee2e96f56`.

Actual smallest-public-movie smoke passed8.422s including I/O, exact graph and
all solver records,99transitions. Only then launched full standalone replay
73943(actualPython5212), now TERMINAL0,428.360s. All4graphs and solver records
match the private-namespace reference, using regenerated raw-input features.
Standalone reportSHA
`81d90e28b45d344e1b30afe195958cf02fcf967d15b3c3a801b2b6cd2694a13f`;
standalone codeSHA
`ffef91dea26d2c12c47d74d7a2846c78a9099b9b62e09afeb8470e1cf16dca4d`.

Important: this compares two implementations of the NEW learned16head, not
the currently submitted fixed8head. Previous Kaggle fixed8event times were
4.549,26.297,30.918,75.576s for these movies. Different CPU hardware makes a
direct ratio unreliable, but the new head is NOT established runtime-safe
under the old hidden-set budget. A fresh platform smoke/projection is required
if any learned candidate passes quality checks. Source44learned head currently
fails raw-edge and worst-movie promotion; this optimization does not rescue
its quality. No draft worker, pending submission, live source6fit or GPU changed.
