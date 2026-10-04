"""Stage only the small new recipe; reuse the immutable remote v2 base bundle."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'.biohub/cache/frozen-image-head-v1-code';output.mkdir(exist_ok=False)
    for source in ('research/frozen_image_head.py','scripts/run-frozen-image-head-v1.py',
                   'reports/experiments/frozen-image-head-v1-design.md'):
        shutil.copyfile(ROOT/source,output/Path(source).name)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    contract=dict(run_id='frozen-image-head-v1',files=files,
        base_bundle_sha256='467076cacc2587444267f6e14f4edb93dd578c27fb1eff76ffbc06cbb290bdb6')
    path=output/'CONTRACT.json';path.write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps(dict(contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files=len(files))))


if __name__=='__main__':main()
