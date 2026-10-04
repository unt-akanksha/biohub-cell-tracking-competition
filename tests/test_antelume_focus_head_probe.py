import ast
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def test_staged_small_archive_integrity_and_scope():
    root=ROOT/'.biohub/cache/antelume-focus-head-replay-v1'
    identity=json.loads((root/'staged_identity.json').read_text());path=root/'probe.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==identity['archive_sha256']
    assert path.stat().st_size<40_000_000
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names=archive.namelist();assert len(names)==len(set(names))==identity['files']
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        manifest=json.loads(archive.read('manifest.json'))
        assert len(manifest['pairs'])==4 and manifest['declared_wall_seconds']==180
        assert {r['stem'] for r in manifest['pairs']}=={'6bba_f1fde7e0','6bba_23af9eeb'}
        assert all(hashlib.sha256(archive.read(p)).hexdigest()==v for p,v in manifest['files'].items())
        source=archive.read('run_probe.py').decode();tree=ast.parse(source)
        assert "--query-compute-apps=pid" in source and "set_per_process_memory_fraction(.2,0)" in source
        assert not any(isinstance(n,ast.Attribute) and n.attr in ('backward','AdamW','kill','terminate') for n in ast.walk(tree))
        assert "if active:raise RuntimeError" in source
