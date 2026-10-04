"""Reuse the pinned conservative one-hour launch checks for the public runtime test."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'scripts/launch-trajectory-event-anchor-kaggle-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest()=='bcabf57f7680c5ca8d88ddf3c02861b904723154ef7f73355fa152f3da46730c'
    source=base.read_text(encoding='utf-8')
    replacements=[
        ('trajectory-event-anchor-kaggle-v1-launch.json','trajectory-event-anchor-production-v1-launch.json'),
        ('trajectory-event-anchor-kaggle-v1-build.json','trajectory-event-anchor-production-v1-build.json'),
        ('biohub-event-anchor-acceptance-v1','biohub-event-anchor-candidate-v1'),
        ("'indarkarhana/biohub-event-anchor-acceptance'","'indarkarhana/biohub-event-anchor-candidate'"),
        ("    metadata=read(stage/'kernel-metadata.json');validate_submission_kernel_metadata(metadata)",
         "    acceptance_path=ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-result.json'\n"
         "    assert sha(acceptance_path)==build['acceptance_sha256']\n"
         "    acceptance=read(acceptance_path)\n"
         "    assert acceptance['status']=='acceptance_passed' and acceptance['production_notebook_eligible']\n"
         "    assert build['only_executable_change_from_acceptance_is_run_mode'] and not build['competition_submission_authorized']\n"
         "    metadata=read(stage/'kernel-metadata.json');validate_submission_kernel_metadata(metadata)"),
        ("    submissions=list(csv.DictReader(",
         "    accepted_status=run(['kernels','status','indarkarhana/biohub-event-anchor-acceptance'])\n"
         "    assert saved_kernel_complete(accepted_status,'indarkarhana/biohub-event-anchor-acceptance')\n"
         "    submissions=list(csv.DictReader("),
        ("quality_failures_preserved=True,submission_performed=False,kernel_pushed=False)",
         "quality_failures_preserved=True,submission_performed=False,kernel_pushed=False,\n"
         "        public_test_only=True,platform_timeout_seconds=3600,\n"
         "        final_submission_version_requires_twelve_hour_timeout_review=True)"),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__))


if __name__=='__main__':main()
