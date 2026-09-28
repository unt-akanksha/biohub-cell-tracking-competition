"""Fixed head-only adaptation settings and a diagnostic screen, not promotion."""

SETTINGS=dict(steps=800,smoke_steps=4,learning_rate=1e-4,weight_decay=1e-4,
              gradient_clip=1.,seed=244691,neural_weight=1.,physical_weight=1.,null_logit=-4.5)


def diagnostic_gate(initial,physical,final):
    import math
    rows=(initial,physical,final)
    if any(not math.isfinite(r['nll']) or r['known_parent']<=0 or r['known_absent']<=0 for r in rows):
        raise ValueError('Finite diagnostics with both real-parent and null labels required')
    if len({(r['known_parent'],r['known_absent']) for r in rows})!=1:
        raise ValueError('Identical diagnostic labels for every control required')
    gates=dict(nll_below_both=final['nll']<min(initial['nll'],physical['nll']),
               parent_correct_not_lower=final['correct_parent']>=max(initial['correct_parent'],physical['correct_parent']),
               null_correct_not_lower=final['correct_absent']>=max(initial['correct_absent'],physical['correct_absent']))
    return dict(passed=all(gates.values()),gates=gates,scope='Training-domain feasibility only; no tracking-score promotion')
