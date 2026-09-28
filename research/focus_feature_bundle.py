"""Validate a generated feature archive before any extraction writes."""
from pathlib import PurePosixPath
import stat

PREFIX='focus_extra_fit_features'


def validate_members(infos,manifest):
    names=[i.filename for i in infos]
    if len(names)!=manifest['files'] or names!=manifest['members'] or len(set(names))!=len(names) or not 1<=len(names)<=1000:
        raise ValueError('Exact unique bounded feature-archive membership required')
    if sum(i.file_size for i in infos)>2_000_000_000:raise ValueError('Feature archive exceeds declared recovery bound')
    for info in infos:
        name=info.filename;parts=PurePosixPath(name).parts
        if ('\\' in name or PurePosixPath(name).is_absolute() or '..' in parts or not parts or parts[0]!=PREFIX
            or any(':' in p for p in parts) or info.is_dir() or stat.S_ISLNK(info.external_attr>>16)):
            raise ValueError('Only regular files inside the exact feature output root are allowed')
        allowed=(parts[1:]==('launcher_terminal.json',)
            or (len(parts)>=3 and parts[1]=='outputs')
            or (len(parts)>=3 and parts[1]=='runtime' and '__pycache__' not in parts and PurePosixPath(name).suffix in ('.py','.json')))
        if not allowed:raise ValueError('Unexpected feature archive artifact')
    return names
