"""The full training collector requires the completed six-movie probe."""
import hashlib
import json

PROBE_SHA='a94fa7229bdd439e7bc10616b52ac4708024bcf39acbc88f346063f919e6d676'


def verify_probe(payload):
    if hashlib.sha256(payload).hexdigest()!=PROBE_SHA:
        raise ValueError('Exact verified small calibration probe required')
    report=json.loads(payload)
    if (report['status']!='verified_training_calibration_collection_probe'
        or report['probe_diagnostic_recall_preserved'] is not True
        or report['authorized_for_submission'] is not False
        or report['full_calibration_completed'] is not False
        or report['fit']['status']!='fitted_training_only_threshold'
        or report['collection']['models_unchanged'] is not True
        or len(report['collection']['records'])!=18):
        raise ValueError('Successful real probe, exact matching and diagnostic recall required')
    return {(r['stem'],r['t']):r for r in report['collection']['records']}


def verify_replay_row(row, expected):
    previous=expected.get((row['stem'],row['t']))
    if previous is None: return False
    for key in ('group','context','annotation_count','input_sha256','parent_peaks','candidate_peaks','artifact'):
        if row[key]!=previous[key]: raise ValueError('Full collection changed probe replay: '+key)
    return True
