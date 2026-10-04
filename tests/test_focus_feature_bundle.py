import stat
import zipfile
import pytest
from research.focus_feature_bundle import validate_members


def check(names):
    infos=[zipfile.ZipInfo(name) for name in names]
    return validate_members(infos,dict(files=len(names),members=names))


def test_required_files_allowed():
    names=['focus_extra_fit_features/launcher_terminal.json','focus_extra_fit_features/outputs/a/000.npz','focus_extra_fit_features/runtime/research/x.py']
    assert check(names)==names


@pytest.mark.parametrize('name',['../x','/tmp/x','focus_extra_fit_features/../x','focus_extra_fit_features\\x',
    'focus_extra_fit_features/repo/LICENSE','focus_extra_fit_features/runtime/__pycache__/x.pyc','focus_extra_fit_features/C:/x'])
def test_unsafe_or_unrelated_members_rejected(name):
    with pytest.raises(ValueError):check([name])


def test_duplicates_and_symlinks_rejected():
    name='focus_extra_fit_features/outputs/result.json'
    with pytest.raises(ValueError):check([name,name])
    info=zipfile.ZipInfo(name);info.external_attr=(stat.S_IFLNK|0o777)<<16
    with pytest.raises(ValueError):validate_members([info],dict(files=1,members=[name]))
