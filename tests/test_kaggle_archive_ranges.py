import io
from pathlib import Path
import runpy
import zipfile
import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'research/kaggle_archive_ranges.py'))


@pytest.mark.parametrize('compression', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_sparse_archive_reader_and_crc_checked_extraction(compression):
    buffer = io.BytesIO()
    name = 'train/example.zarr/0/c/0/0/0/0'; content = b'raw chunk' * 1000
    with zipfile.ZipFile(buffer, 'w', compression=compression) as file:
        file.writestr(name, content)
    data = buffer.getvalue(); fetch = lambda a, b: data[a:b + 1]
    reader = M['RangeReader'](len(data), fetch)
    with zipfile.ZipFile(reader) as file:
        info = file.getinfo(name)
    record = dict(path=name, header_offset=info.header_offset, compress_type=info.compress_type,
                  bytes=info.file_size, compressed_bytes=info.compress_size, crc32=info.CRC)
    assert M['extract_member'](record, fetch) == content
    record['crc32'] ^= 1
    with pytest.raises(ValueError, match='CRC'):
        M['extract_member'](record, fetch)


def test_whole_large_archive_read_is_refused():
    reader = M['RangeReader'](87_000_000_000, lambda a, b: b'')
    with pytest.raises(ValueError, match='unbounded'):
        reader.read()
