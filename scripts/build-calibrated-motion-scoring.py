"""CPU official scoring of immutable calibrated inference artifacts."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-calibrated-motion-scoring-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-image-motion-linker-scoring.py'))['build']()
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    bundle['selection_launch.ipynb'] = (ROOT/'kaggle/biohub-calibrated-motion-selection-v1/biohub-calibrated-motion-selection-v1.ipynb').read_text()
    tail = source[source.index('\nimport runpy'):].replace('biohub-image-motion-linker-selection-v1','biohub-calibrated-motion-selection-v1')
    tail = tail.replace("p/'image_motion_linker_selection'","p/'calibrated_motion_selection'")
    nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    source = ''.join(nb['cells'][0]['source']).replace('image_motion_linker_score','calibrated_motion_score')
    source = source.replace('image-motion-linker-scoring-v1','calibrated-motion-scoring-v1')
    nb['cells'][0]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex']['run_id']='calibrated-motion-scoring-v1'
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Calibrated Motion Scoring v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-calibrated-motion-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
