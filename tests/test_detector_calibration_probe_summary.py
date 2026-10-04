from pathlib import Path
import runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/summarize-detector-calibration-probe.py'))


def test_probe_cutoff_replays_exact_matching_without_deployment_authority():
    rows=[]
    for group in ('fitting','diagnostic'):
        truth=np.array([[1,0,0,0],[1,0,50,50]])
        rows.append((dict(stem=group,t=1,group=group),dict(truth_coords=truth,
            parent_coords=truth[:1],parent_probabilities=np.array([.9],dtype=np.float32),
            candidate_coords=truth,candidate_probabilities=np.array([.9,.7],dtype=np.float32))))
    report=M['evaluate_records'](rows,.5)
    assert report['probe_diagnostic_recall_preserved']
    assert all(r['candidate_calibrated_matched']==1 for r in report['per_frame'])
    assert not report['authorized_for_submission'] and not report['full_calibration_completed']
    rows[-1][1]['candidate_probabilities'][:]=.6
    report=M['evaluate_records'](rows,.5)
    assert not report['probe_diagnostic_recall_preserved']
