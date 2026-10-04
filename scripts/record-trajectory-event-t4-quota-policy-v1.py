"""Capture a direct Kaggle staff clarification, not a community inference."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'reports/experiments/trajectory-event-t4-quota-policy-v1.json'
    assert not output.exists()
    from kaggle import api
    topic,comments,_=api.forums_topic_show(361104)
    stack=list(comments);found=[]
    while stack:
        comment=stack.pop();stack.extend(comment.replies or [])
        if comment.id==1995818:found.append(comment)
    assert len(found)==1
    comment=found[0];assert comment.author_name=='Dustin'
    excerpt='1 hour of Notebook runtime in the 2xT4 setup only consumes 1 hour of your GPU quota'
    assert excerpt in comment.content
    result=dict(status='direct_staff_t4_quota_clarification_verified',
        checked_utc=datetime.now(timezone.utc).isoformat(),topic_id=361104,comment_id=1995818,
        author_name=comment.author_name,author_url=comment.author_url,
        source_url='https://www.kaggle.com/discussions/product-feedback/361104#1995818',
        excerpt=excerpt,comment_content_sha256=hashlib.sha256(comment.content.encode()).hexdigest(),
        t4x2_quota_hours_per_notebook_hour=1,
        corroborating_current_source='https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators',
        pending_hidden_evaluation_assumed_free=False,pending_platform_cap_reserved_hours=12,
        new_one_hour_acceptance_reserved_hours=2,minimum_unreserved_hours=8,
        retrospective_receipts_not_changed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
