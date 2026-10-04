"""Verify every division experiment artifact against a remote hash manifest."""
import hashlib
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SSH = ['-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128']
HOST = 'ubuntu@3.226.249.134'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--proposal-audit-v4',action='store_true');mode.add_argument('--additive-data-v5',action='store_true')
    mode.add_argument('--additive-heads-v5',action='store_true');mode.add_argument('--context-v6',action='store_true');args=parser.parse_args()
    groups = {
        'native-division-v3-feature-full': '/dev/shm/biohub-native-division-v3-feature-full',
        'native-division-v3-feature-smoke': '/dev/shm/biohub-native-division-v3-feature-smoke',
        'native-division-v3-head-full': '/tmp/biohub-image-context-v2.ScdSdY/native-division-v3-head-full',
        'native-division-v3-head-smoke': '/tmp/biohub-image-context-v2.ScdSdY/native-division-v3-head-smoke',
    }
    if args.proposal_audit_v4:
        groups={name:'/dev/shm/biohub-'+name for name in ('native-division-proposal-audit-v4-smoke','native-division-proposal-audit-v4-full')}
    if args.additive_data_v5:
        groups={name:'/dev/shm/biohub-'+name for name in ('native-division-additive-v5-smoke-r2','native-division-additive-v5-full')}
    if args.additive_heads_v5:
        groups={name:'/dev/shm/biohub-'+name for name in ('native-division-additive-v5-feature-full','native-division-additive-v5-feature-smoke','native-division-additive-v5-features-merged')}
        groups.update({name:'/tmp/biohub-image-context-v2.ScdSdY/'+name for name in ('native-division-additive-v5-head-smoke','native-division-additive-v5-head-full')})
    if args.context_v6:
        groups={name:'/dev/shm/biohub-'+name for name in ('native-division-context-v6-smoke','native-division-context-v6-full','native-division-context-v6-model-smoke')}
    code = '''import json,hashlib
from pathlib import Path
groups=''' + repr(groups) + '''
out={}
for name,folder in groups.items():
 root=Path(folder).resolve(strict=True)
 assert json.loads((root/'RESULT.json').read_text())['status'] in ('features_complete','feature_smoke_passed','division_head_screen_complete','head_smoke_passed','proposal_audit_smoke_passed','proposal_audit_complete','additive_smoke_passed','additive_extraction_complete','context_smoke_passed','context_data_complete','temporal_model_smoke_passed')
 out[name]=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file()]
 assert all(not p.is_symlink() for p in root.rglob('*'))
print(json.dumps(out))
'''
    response = subprocess.run(['ssh', *SSH, HOST, '/home/ubuntu/venv/bin/python -c ' + shlex.quote(code)],
                              capture_output=True, text=True, check=True, timeout=60)
    manifests = json.loads(response.stdout)
    receipts = []
    for name, remote in groups.items():
        local = ROOT / '.biohub/cache' / (name + '-output')
        if local.exists():
            raise ValueError('Do not overwrite an existing backup')
        subprocess.run(['scp', *SSH, '-r', HOST + ':' + remote, str(local)], check=True, timeout=180)
        for record in manifests[name]:
            path = local / record['path']
            if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                raise ValueError('Artifact backup mismatch')
        if {p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()} != {r['path'] for r in manifests[name]}:
            raise ValueError('Backup file set changed')
        shutil.copy2(local / 'RESULT.json', ROOT / 'reports/experiments' / (name + '-result.json'))
        receipts.append(dict(name=name, local_root=str(local), remote_root=remote, records=manifests[name]))
    result = dict(status='verified_backup', groups=receipts, remote_files_deleted=False)
    receipt='native-division-proposal-audit-v4-harvest.json' if args.proposal_audit_v4 else 'native-division-v3-harvest.json'
    if args.additive_data_v5:receipt='native-division-additive-v5-data-harvest.json'
    if args.additive_heads_v5:receipt='native-division-additive-v5-head-harvest.json'
    if args.context_v6:receipt='native-division-context-v6-harvest.json'
    (ROOT / 'reports/experiments' / receipt).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], groups=len(receipts),
                         bytes=sum(r['bytes'] for group in receipts for r in group['records']))))


if __name__ == '__main__':
    main()
