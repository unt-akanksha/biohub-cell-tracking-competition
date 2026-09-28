# Microscopy checkpoint screen, September10

No new model weights downloaded or executed. Active GPU work remains the owned
training-confidence collector. This screen does not establish Biohub accuracy.

Cellpose's official repository reports June2026 additions: DINOv3-based cpdino
and cpdino-vitb, plus cpsam_v2 aimed at reducing low-contrast spurious masks.
These are distinct from the existing point-detector family. However, the same
README warns that all its models use CC-BY-NC training data. The code is BSD,
and the official Hugging Face model repository labels its model BSD-3-Clause
but has no substantive model-card text. Do not conflate code, weight metadata
and dataset permissions. Their compatibility with this workspace's commercial
reproducibility requirements remains unresolved; not approved for integration
or GPU spending on the evidence inspected. This is a conservative project
screen, not a legal conclusion that every use of the weights is prohibited.
[Official repository](https://github.com/MouseLand/cellpose),
[code license](https://github.com/MouseLand/cellpose/blob/main/LICENSE),
[official model repository](https://huggingface.co/mouseland/cellpose-sam).

Cellpose's documented volumetric method predicts orthogonal planar flows and
combines them for3D mask dynamics. Biohub adaptation would require validated
mask-to-center conversion, physical-axis handling, recall and runtime profiling;
2D screenshots or segmentation claims would not establish tracking quality.
[Official3D documentation](https://cellpose.readthedocs.io/en/latest/do3d.html).

StarDist's current FAQ says a general pretrained3D model is unavailable, while
the implementation registers a3D_demo checkpoint. Treat the latter as a demo,
not evidence of a broadly pretrained, independently strong Biohub detector.
No StarDist weights were downloaded, and training-data/weight licensing was
not established by this screen.
[Official FAQ](https://github.com/stardist/stardist-docs/blob/main/docs/faq.md),
[registered models](https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py).

Decision: keep these as unapproved research leads; no speculative expensive
model migration. Complete the already-running owned-model calibration test.
