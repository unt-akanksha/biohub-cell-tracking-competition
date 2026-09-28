# FOCUS fixed-division ordinary-link assignment

CPU-only source-development comparison, declared before candidate scoring.
The previous detector/flow candidate gained0.1454931 over parent but failed
the per-movie regression limit. Its67ebd073 static detector arm already
regresses; association is only one possible contributor, not a guaranteed fix.

Freeze all FOCUS source detections and sampled native flow from completed
source-flow version1, notebook d36fd6ef5fb460ae4964429aded2832b0d44db91a65836476ecea64639f30375.
Reference report SHA896d3de86fa30fcd860f2c16073d47718118d94e1a43fe70710626c7b32ea169.
Keep every existing two-child fork and both children locked. Apply rectangular
one-to-one linear assignment to the remaining next-frame nodes, with one
private null per child. Reuse source-trained variance and null cost4.5;
reject exact-null ties. No cost sweep, node pruning, count calibration,
new divisions, gap links, image inference or retraining. Unlike the previous
ordinary LAP experiment, this does not suppress all divisions. That older
experiment used different detector nodes and was not promoted.

Small tests must pass before processing the fixed8 source movies. Verify
original source artifacts/graphs, persist all8 new graphs before GT, replay
every control score exactly, and report all movie/embryo metrics. Source
movies are already exposed development data; no new target data may open.

Acceptance requires the unchanged source-vs-parent gate, plus positive score
and raw-edge gains over the new FOCUS-flow reference, no movie loss greater
than0.02 versus that reference, and at least its3 true positive divisions.
The original parent gate already requires >=5/8 gains, per-movie loss<=0.02,
worst-movie minimum loss<=0.01 and recall loss<=0.005. No retrospective gate
relaxation. Even success is not submission authorization or public-best proof.
FOCUS pretraining overlap remains unverified. No GPU job is launched.
