"""Bounded, ETag-pinned read-only access to authorized Kaggle archive ranges."""
import io
import struct
import zlib


class RangeReader(io.RawIOBase):
    def __init__(self, length, fetch):
        self.length, self.fetch, self.position = int(length), fetch, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError('Invalid archive seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = max(0, self.length - self.position) if size < 0 else min(size, max(0, self.length - self.position))
        if size > 32 * 1024 ** 2:
            raise ValueError('Refuse unbounded archive download')
        if not size:
            return b''
        data = self.fetch(self.position, self.position + size - 1)
        if len(data) != size:
            raise ValueError('Incomplete archive range')
        self.position += size
        return data


def extract_member(record, fetch):
    offset = record['header_offset']
    header = fetch(offset, offset + 29)
    values = struct.unpack('<4s5H3I2H', header)
    signature, flags, method = values[0], values[2], values[3]
    name_len, extra_len = values[-2:]
    if signature != b'PK\x03\x04' or flags & 1 or method != record['compress_type']:
        raise ValueError('Unsupported or changed ZIP header')
    name = fetch(offset + 30, offset + 29 + name_len).decode('utf-8')
    if name != record['path'] or method not in (0, 8):
        raise ValueError('Wrong archive member')
    size, packed = record['bytes'], record['compressed_bytes']
    if not (0 < size <= 16 * 1024 ** 2 and 0 < packed <= 16 * 1024 ** 2):
        raise ValueError('Unexpected movie chunk size')
    start = offset + 30 + name_len + extra_len
    data = fetch(start, start + packed - 1)
    if method == 8:
        decoder = zlib.decompressobj(-15)
        decoded = decoder.decompress(data, size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid or oversized compressed member')
        data = decoded
    if len(data) != size or zlib.crc32(data) != record['crc32']:
        raise ValueError('Archive member size/CRC mismatch')
    return data
