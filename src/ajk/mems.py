from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from collections.abc import Iterator

import numpy as np
import pymem

PROT_READABLE = (0x02, 0x04, 0x20, 0x40, 0x80)
CHUNK = 1 << 22


class MBI(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


_kernel32 = ctypes.windll.kernel32
_kernel32.VirtualQueryEx.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_kernel32.VirtualQueryEx.restype = ctypes.c_size_t


def iterate_regions(
    pm: pymem.Pymem,
    private_only: bool = True,
) -> Iterator[tuple[int, int]]:
    mbi = MBI()
    address = 0
    handle = ctypes.c_void_p(pm.process_handle)
    while address < 0x7FFFFFFF:
        if not _kernel32.VirtualQueryEx(
            handle,
            ctypes.c_void_p(address),
            ctypes.cast(ctypes.pointer(mbi), ctypes.c_void_p),
            ctypes.sizeof(mbi),
        ):
            break
        base = int(mbi.BaseAddress or 0)
        size = int(mbi.RegionSize or 0)
        state = int(mbi.State)
        protect = int(mbi.Protect)
        mtype = int(mbi.Type)
        if state == 0x1000 and protect in PROT_READABLE:
            if not private_only or mtype == 0x20000:
                yield base, size
        if size == 0:
            break
        address = base + size


def _read_chunks(pm: pymem.Pymem, base: int, size: int) -> Iterator[tuple[int, bytes]]:
    offset = 0
    while offset < size:
        n = min(CHUNK, size - offset)
        try:
            data = pm.read_bytes(base + offset, n)
        except Exception:
            offset += CHUNK
            continue
        yield base + offset, data
        offset += CHUNK


def scan_floats(
    pm: pymem.Pymem,
    lo: float,
    hi: float,
    private_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    addrs: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for base, size in iterate_regions(pm, private_only):
        for chunk_base, data in _read_chunks(pm, base, size):
            arr = np.frombuffer(data, dtype=np.float32)
            idx = np.flatnonzero((arr >= lo) & (arr <= hi))
            if idx.size:
                addrs.append(chunk_base + idx.astype(np.int64) * 4)
                vals.append(arr[idx])
    if not addrs:
        return np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(addrs), np.concatenate(vals)


def scan_float_pairs(
    pm: pymem.Pymem,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    private_only: bool = True,
    max_results: int = 4000,
) -> list[tuple[int, float, float]]:
    found: list[tuple[int, float, float]] = []
    for base, size in iterate_regions(pm, private_only):
        for chunk_base, data in _read_chunks(pm, base, size):
            usable = len(data) - (len(data) % 4)
            if usable < 8:
                continue
            arr = np.frombuffer(data[:usable], dtype=np.float32)
            xs = arr[:-1]
            ys = arr[1:]
            mask = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
            idx = np.flatnonzero(mask)
            for i in idx.tolist():
                found.append((int(chunk_base + i * 4), float(xs[i]), float(ys[i])))
                if len(found) >= max_results:
                    return found
    return found


def scan_pointers(
    pm: pymem.Pymem,
    targets: list[int] | np.ndarray,
    private_only: bool = False,
) -> list[tuple[int, int]]:
    target_arr = np.asarray(sorted(set(int(t) for t in targets)), dtype=np.uint32)
    if target_arr.size == 0:
        return []
    hits: list[tuple[int, int]] = []
    for base, size in iterate_regions(pm, private_only):
        for chunk_base, data in _read_chunks(pm, base, size):
            usable = len(data) - (len(data) % 4)
            if usable <= 0:
                continue
            arr = np.frombuffer(data[:usable], dtype=np.uint32)
            mask = np.isin(arr, target_arr)
            idx = np.flatnonzero(mask)
            for i in idx:
                hits.append((chunk_base + int(i) * 4, int(arr[i])))
    return hits


def read_floats(pm: pymem.Pymem, addr: int) -> tuple[float, float] | None:
    try:
        data = pm.read_bytes(addr, 8)
    except Exception:
        return None
    if len(data) < 8:
        return None
    x, y = np.frombuffer(data, dtype=np.float32)
    return float(x), float(y)
