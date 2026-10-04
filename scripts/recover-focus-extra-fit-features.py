"""Recover the completed feature bundle; model verification remains separate."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_feature_bundle import PREFIX,validate_members


if __name__=='__main__':
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1'
    archive_path=folder/'focus_extra_fit_features_bundle.zip';manifest_path=archive_path.with_suffix('.json')
    manifest=json.loads(manifest_path.read_text())
    if hashlib.sha256(archive_path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Actual bundle digest differs')
    destination=folder/PREFIX
    if destination.exists():raise ValueError('Do not overwrite an existing feature extraction; verify it separately')
    with zipfile.ZipFile(archive_path) as archive:
        names=validate_members(archive.infolist(),manifest)
        root=folder.resolve()
        if any(not (folder/name).resolve().is_relative_to(root) for name in names):raise ValueError('Extraction would escape exact cache directory')
        if archive.testzip() is not None:raise ValueError('Feature archive CRC failure')
        archive.extractall(folder)
    print(json.dumps(dict(status='feature_bundle_recovered_not_yet_model_verified',files=len(names),sha256=manifest['sha256'])))
