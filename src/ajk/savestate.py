"""Exact player state from the game's own SaveState struct (memory).

The game writes a 32-byte SaveState to the static field
`SaveLube._COMBINED_SAVE.player_position` every frame:

    offset 0  float pos_x     (world x, clamped to [9, 471])
    offset 4  float pos_y     (world y, negative going up the tower)
    offset 8  float vel_x     (px per frame)
    offset 12 float vel_y     (px per frame)
    offset 16 int   state     (0 normal, 1 knocked, 2 splat)
    offset 20 int   time_stamp (increments once per game frame)
    offset 24 bool  is_on_ground
    offset 28 int   direction (1 or -1)

Because it lives in static storage the address never moves; it is found by a
vectorised scan and verified by watching `time_stamp` tick.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass

import numpy as np
import pymem

from .mems import _read_chunks, iterate_regions

SAVE_SIZE = 32
GROUNDED_OFFSET = 24
TIMESTAMP_OFFSET = 20
VELOCITY_OFFSET = 8


@dataclass
class PlayerSaveState:
    address: int = 0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    state: int = 0
    time_stamp: int = 0
    grounded: bool = True
    direction: int = 1
    when: float = 0.0

    @property
    def screen_index(self) -> int:
        return screen_index_for_y(self.y)


def screen_index_for_y(y: float) -> int:
    """The game's camera index: -floor(centerY / 360) with centerY = y + 13."""
    return -int(np.floor((y + 13.0) / 360.0))


def _candidates_from_chunk(
    chunk_base: int,
    data: bytes,
    near_x: float | None = None,
    near_y: float | None = None,
) -> list[int]:
    usable = len(data) - (len(data) % 4)
    if usable < SAVE_SIZE:
        return []
    arr = np.frombuffer(data[:usable], dtype=np.float32)
    if arr.size < 8:
        return []
    x = arr[:-7]
    y = arr[1:-6]
    vx = arr[2:-5]
    vy = arr[3:-4]
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(vx)
        & np.isfinite(vy)
        & (x >= -20.0)
        & (x <= 500.0)
        & (y >= -61000.0)
        & (y <= 800.0)
        & (vx >= -6.0)
        & (vx <= 6.0)
        & (vy >= -12.0)
        & (vy <= 12.0)
    )
    if near_x is not None and near_y is not None and np.isfinite(near_x) and np.isfinite(near_y):
        with np.errstate(invalid="ignore"):
            mask &= np.abs(x - near_x) <= 30.0
            delta = np.abs(y - near_y) % 360.0
            mask &= np.minimum(delta, 360.0 - delta) <= 30.0
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    ints = np.frombuffer(data[:usable], dtype=np.int32)
    raw = np.frombuffer(data, dtype=np.uint8)
    state = ints[idx + 4]
    stamp = ints[idx + 5]
    direction = ints[idx + 7]
    ground = raw[idx * 4 + GROUNDED_OFFSET]
    ok = (
        ((state == 0) | (state == 1) | (state == 2))
        & (stamp > 0)
        & (stamp < 2_000_000_000)
        & ((ground == 0) | (ground == 1))
        & ((direction == -1) | (direction == 1))
    )
    return [int(chunk_base + int(i) * 4) for i in idx[ok]]


def read_save_state(pm: pymem.Pymem, address: int) -> PlayerSaveState | None:
    try:
        data = pm.read_bytes(address, SAVE_SIZE)
    except Exception:
        return None
    if len(data) < SAVE_SIZE:
        return None
    x, y, vx, vy = struct.unpack_from("<ffff", data, 0)
    state, stamp = struct.unpack_from("<ii", data, 16)
    grounded = data[GROUNDED_OFFSET] != 0
    direction = struct.unpack_from("<i", data, 28)[0]
    if not all(math.isfinite(v) for v in (x, y, vx, vy)):
        return None
    # Denormal positions are reinterpreted garbage (seen: unrelated 60 Hz
    # counter structs pass the range checks); real coordinates never are.
    if any(0.0 < abs(v) < 1e-6 for v in (x, y)):
        return None
    if not (-20.0 <= x <= 500.0 and -61000.0 <= y <= 800.0):
        return None
    if not (-12.0 <= vy <= 12.0) or not (-6.0 <= vx <= 6.0):
        return None
    if state not in (0, 1, 2) or stamp <= 0:
        return None
    # PlayerEntity.GetSaveState only ever writes -1 or 1.
    if direction not in (-1, 1):
        return None
    return PlayerSaveState(
        address=address,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        state=state,
        time_stamp=stamp,
        grounded=grounded,
        direction=direction,
        when=time.monotonic(),
    )


def _scan_candidates(
    pm: pymem.Pymem,
    near_x: float | None = None,
    near_y: float | None = None,
) -> list[int]:
    candidates: list[int] = []
    for base, size in iterate_regions(pm, private_only=True):
        for chunk_base, data in _read_chunks(pm, base, size):
            candidates.extend(_candidates_from_chunk(chunk_base, data, near_x, near_y))
    return candidates


def _phase_delta(value: float, reference: float, period: float = 360.0) -> float:
    delta = abs(value - reference) % period
    return min(delta, period - delta)


def _tick_test(
    pm: pymem.Pymem,
    addresses: list[int],
    window_s: float = 0.09,
    require_position: bool = False,
) -> int | None:
    first = {addr: read_save_state(pm, addr) for addr in addresses}
    time.sleep(window_s)
    best: tuple[int, int, int] | None = None
    for addr, before in first.items():
        if before is None:
            continue
        if require_position and before.x == 0.0 and before.y == 0.0:
            continue
        after = read_save_state(pm, addr)
        if after is None:
            continue
        delta = after.time_stamp - before.time_stamp
        if 1 <= delta <= 40:
            score = (after.time_stamp, -abs(delta - 6), addr)
            if best is None or score[:2] > best[:2]:
                best = score
    return best[2] if best is not None else None


def find_save_state(
    pm: pymem.Pymem,
    near_x: float | None = None,
    near_y: float | None = None,
    timeout_s: float = 20.0,
) -> int | None:
    """Scan memory for the SaveState struct and verify that its timestamp ticks.

    With a vision hint the candidate list is filtered by position (the y is
    compared modulo 360 because the screen index is unknown), which keeps the
    verification reads close enough together in time to see the tick.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if near_x is not None and near_y is not None:
            filtered = _scan_candidates(pm, near_x, near_y)
            if filtered:
                found = _tick_test(pm, filtered, require_position=True)
                if found is not None:
                    return found
            continue
        candidates = _scan_candidates(pm)
        if candidates:
            found = _tick_test(pm, candidates[:60000], require_position=True)
            if found is not None:
                return found
    return None


def debug_candidates(pm: pymem.Pymem) -> list[tuple[int, PlayerSaveState]]:
    candidates = _scan_candidates(pm)
    first = {addr: read_save_state(pm, addr) for addr in candidates}
    ticking: list[tuple[int, PlayerSaveState]] = []
    for addr, before in first.items():
        if before is None:
            continue
        after = read_save_state(pm, addr)
        if after is None:
            continue
        if 1 <= after.time_stamp - before.time_stamp <= 40:
            ticking.append((addr, after))
    return ticking


class PlayerMemoryV2:
    """Reads the exact player state, re-scanning when the address goes bad."""

    def __init__(self, process_name: str = "JumpKing.exe") -> None:
        self.process_name = process_name
        self.address: int | None = None
        self._pm: pymem.Pymem | None = None
        self._last: PlayerSaveState | None = None
        self._last_read_at = 0.0
        self._failures = 0
        self.attached = False

    def attach(self) -> bool:
        try:
            self._pm = pymem.Pymem(self.process_name)
            self.attached = True
        except Exception:
            self._pm = None
            self.attached = False
        return self.attached

    @property
    def pm(self) -> pymem.Pymem | None:
        return self._pm

    def ensure_address(
        self,
        near_x: float | None = None,
        near_y: float | None = None,
        timeout_s: float = 20.0,
    ) -> bool:
        if self._pm is None and not self.attach():
            return False
        if self.address is not None:
            state = read_save_state(self._pm, self.address)
            if state is not None and (state.x != 0.0 or state.y != 0.0):
                return True
        if self._pm is None:
            return False
        found = find_save_state(self._pm, near_x, near_y, timeout_s=timeout_s)
        self.address = found
        return found is not None

    def read(self) -> PlayerSaveState | None:
        if self._pm is None or self.address is None:
            return None
        state = read_save_state(self._pm, self.address)
        if state is None:
            self._failures += 1
            if self._failures >= 5:
                self.address = None
            return None
        self._failures = 0
        self._last = state
        self._last_read_at = time.monotonic()
        return state

    @property
    def last(self) -> PlayerSaveState | None:
        return self._last
