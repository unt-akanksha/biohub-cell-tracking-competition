"""Read-only recent submission audit; scores are not model-selection inputs."""
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi(); api.authenticate()
    rows=api.competition_submissions('biohub-cell-tracking-during-development',page_size=20) or []
    records=[]
    for row in rows:
        if row is None: continue
        value=dict(ref=getattr(row,'ref',None),file_name=getattr(row,'file_name',None),
            date=str(getattr(row,'date',None)),description=getattr(row,'description',None),
            status=str(getattr(row,'status',None)),public_score=getattr(row,'public_score',None))
        records.append(value)
    if not records: raise ValueError('No authoritative submission records returned')
    result=dict(status='read_only_recent_submission_audit',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        page_size=20,records=records,model_selection_from_public_scores=False,
        submissions_created=False,kernels_modified=False,
        caveat='Missing scores are unknown, not zero or a notebook-title score. This date-ordered history audit does not promote a model or execute any older notebook.')
    target=ROOT/'reports/experiments/recent-submission-audit-20260910.json'
    if target.exists(): raise ValueError('Refuse to overwrite completed submission snapshot')
    target.write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(dict(status=result['status'],count=len(records),latest=records[0]),indent=2,default=str))


if __name__=='__main__': main()
