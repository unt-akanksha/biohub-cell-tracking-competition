# Fitting features retain signal; centering is not supported

CPU-only audit completed21.25s on2,818 uniquely annotated parent comparisons
from the four fitting movies. Each alternative is the closest incorrect source
under the fixed physical prior. No diagnostic representation statistics,
optimizer, predictions, source-selection or target access.

| Simple pairwise rule | True-parent preference |
| --- | ---: |
| Raw feature cosine | 79.67% |
| Frame-pair centered cosine | 73.53% |
| Feature L2 distance | 76.15% |
| Physical prior | 91.66% |

Mean true-parent cosine0.999562 versus incorrect-parent0.997424. Therefore the
high average cosine from the presence fit is not by itself evidence of feature
collapse. Raw cosine has useful discrimination on this fitting diagnostic;
centering worsens all four movies and is not adopted. None of these pairwise
statistics evaluates the learned transformer or proves tracking improvement.

Two focused tests pass. Actual resultSHA256:
`acb68b3fdcae88e257c06d8782648cc7ded55438ba07f9c7d0daf150d9d1349b`.
All original feature/cache hashes verified before the audit. No GPU consumed.
