from __future__ import annotations

import struct
from dataclasses import dataclass

import lz4.block
import numpy as np

SURFACE_FORMATS: dict[int, tuple[str, int]] = {
    0: ("Color", 4),
    1: ("Bgr565", 2),
    2: ("Bgra5551", 2),
    3: ("Bgra4444", 2),
    4: ("Dxt1", 0),
    5: ("Dxt3", 0),
    6: ("Dxt5", 0),
    12: ("Alpha8", 1),
    13: ("Single", 4),
    17: ("HalfVector2", 4),
    18: ("HalfVector4", 8),
}


@dataclass
class XnbFile:
    platform: str
    version: int
    flags: int
    size: int
    readers: list[str]
    payload: bytes
    payload_pos: int


def _read_7bit(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _read_string(data: bytes, pos: int) -> tuple[str, int]:
    length, pos = _read_7bit(data, pos)
    raw = data[pos : pos + length]
    return raw.decode("utf-8", errors="replace"), pos + length


def parse(data: bytes) -> XnbFile:
    if data[:3] != b"XNB":
        raise ValueError("not an XNB file")
    platform = chr(data[3])
    version = data[4]
    flags = data[5]
    size = struct.unpack_from("<I", data, 6)[0]
    pos = 10
    if flags & 0x80:
        decompressed_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        payload = lz4.block.decompress(data[pos:], uncompressed_size=decompressed_size)
    else:
        payload = data[pos:]
    p = 0
    n_readers, p = _read_7bit(payload, p)
    readers: list[str] = []
    for _ in range(n_readers):
        name, p = _read_string(payload, p)
        p += 4
        readers.append(name)
    n_shared, p = _read_7bit(payload, p)
    for _ in range(n_shared):
        _ = struct.unpack_from("<I", payload, p)[0]
        p += 4
    return XnbFile(platform, version, flags, size, readers, payload, p)


def read_object_header(xnb: XnbFile, pos: int) -> tuple[int, int]:
    type_index, pos = _read_7bit(xnb.payload, pos)
    return type_index, pos


def read_texture(xnb: XnbFile, pos: int) -> tuple[np.ndarray, int]:
    fmt, width, height, mips = struct.unpack_from("<IIII", xnb.payload, pos)
    pos += 16
    name, bpp = SURFACE_FORMATS.get(fmt, (f"unknown({fmt})", 0))
    if bpp == 0:
        raise NotImplementedError(f"surface format {name} not supported")
    n_bytes = width * height * bpp
    raw = xnb.payload[pos : pos + n_bytes]
    pos += n_bytes
    if fmt == 0:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        img = arr[:, :, [2, 1, 0, 3]].copy()
    elif fmt == 12:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        img = np.dstack([arr, arr, arr, np.full_like(arr, 255)])
    else:
        raise NotImplementedError(f"decoding {name} not implemented")
    return img, pos
