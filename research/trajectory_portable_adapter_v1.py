"""Reproducible path/scope adapter for the already validated inference worker.

No predictor, model, postprocessor or source-expert mathematics are rewritten.
The builder pins the original worker before calling this mechanical adapter.
"""
from __future__ import annotations

import ast


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one portability anchor: {old[:80]}')
    return source.replace(old, new, 1)


def portable_source(source):
    start = source.index("    helper = load_module(")
    end = source.index("    motion = load_module(", start)
    source = source[:start] + '''    from trajectory_runtime_v1 import verify_bundle, image_metadata
    contract = verify_bundle(args.bundle, args.contract_sha256)
    helper = load_module(args.bundle / 'trajectory_division_full_movie_v1.py', 'd4_full_helper')
    helper.image_metadata = image_metadata
    pre = load_module(args.bundle / 'public_d4_preflight.py', 'd4_preflight_helper')
''' + source[end:]
    source = replace_once(source, "    import tracksdata as td", """    import tracksdata as td
    import ilpy
    # Antelume used SCIP. Pin that same backend even if Kaggle has Gurobi.
    import tracksdata.solvers._ilp_solver as td_ilp
    original_solver = ilpy.Solver
    class ScipSolver(original_solver):
        def __init__(self, *a, **kw):
            kw['preference'] = ilpy.Preference.Scip
            super().__init__(*a, **kw)
    td_ilp.Solver = ScipSolver""")
    source = replace_once(source, 'One freed Antelume CUDA device required',
                          'Exactly one isolated visible CUDA device per worker required')
    source = replace_once(source, "                  independently_held_out=False, kaggle_gpu_hours=0)",
                          "                  independently_held_out=False, solver_backend='SCIP')")
    source = replace_once(source, "    frames = 8 if args.mode == 'smoke' else 100\n    stems = helper.STEMS[:1] if args.mode == 'smoke' else helper.STEMS",
                          "    stems = args.movies")
    source = replace_once(source, '    for stem in stems:\n',
                          "    for stem in stems:\n        frames = 8 if args.mode == 'smoke' else image_metadata(args.images / (stem+'.zarr')).image_shape[0]\n")
    if source.count("args.images / 'train'") != 2:
        raise ValueError('Expected exactly two image-root path replacements')
    source = source.replace("args.images / 'train'", 'args.images')
    # Core inference/ILP/repair block must remain byte-for-byte unchanged except
    # its explicit image-root replacement, verified independently in tests.
    source = source[:source.index('\ndef main():')]
    source += '''
def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'images', 'output', 'movies-json'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--contract-sha256', required=True)
    p.add_argument('--mode', choices=('smoke', 'full'), required=True)
    p.add_argument('--wall-cap-seconds', type=int, required=True)
    args = p.parse_args()
    args.movies = json.loads(args.movies_json.read_text())
    from trajectory_runtime_v1 import validate_movie_ids
    validate_movie_ids(args.movies)
    if not 0 < args.wall_cap_seconds <= 36000:
        raise ValueError('Invalid worker time budget')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(run_id='trajectory-portable-worker-v1', status='running',
                  authorized_for_submission=False)
    def timeout():
        (args.output / 'timeout.json').write_text(json.dumps(dict(status='timeout')))
        os._exit(124)
    timer = threading.Timer(args.wall_cap_seconds, timeout)
    timer.daemon = True
    timer.start()
    try:
        execute(args, result)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        timer.cancel()
        result['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
'''
    ast.parse(source)
    return source
