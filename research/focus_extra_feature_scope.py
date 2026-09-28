"""Feature extraction scope for new fitting movies; no diagnostics extracted."""
from research.focus_extra_fit_scope import scope as raw_scope


def scope(payload):
    return dict(raw_scope(payload),diagnostic_stems=[],
        feature_scope='Eight new fitting movies only; original four diagnostic summaries reused unchanged')
