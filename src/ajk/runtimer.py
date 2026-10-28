"""The game's attempt timer: `AchievementManager.GetCurrentStats().timeSpan`.

`WindManager.CurrentVelocityRaw` derives the wind phase from
`AchievementManager.instance.GetCurrentStats().timeSpan`, i.e. the all-time
stats minus the attempt snapshot:

    ticks_delta = m_all_time_stats._ticks - m_snapshot._ticks
    seconds     = ticks_delta * TargetElapsedTime.TotalSeconds   (1/60)

`PlayerStats` is a 28-byte struct: float time; int jumps, falls, attempts,
session, times_won, _ticks. The manager stores the snapshot first, then the
all-time stats, so both tick counters are 28 bytes apart in memory.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np
import pymem

from .mems import _read_chunks, iterate_regions

STATS_SIZE = 28
TICKS_OFFSET = 24
MAX_TICK_CANDIDATES = 30000


def _plausible_stats(prev: np.ndarray, ticks: np.ndarray, snap: np.ndarray) -> np.ndarray:
    """`prev` is the int32 array of the six fields before `_ticks` for each
    candidate, ordered times_won, session, attempts, falls, jumps."""
    times_won = prev[0]
    session = prev[1]
    attempts = prev[2]
    falls = prev[3]
    jumps = prev[4]
    return (
        (ticks > 600)
        & (times_won >= 0)
        & (times_won < 100_000)
        & (session >= 1)
        & (session < 100_000)
        & (attempts >= 1)
        & (attempts < 100_000)
        & (falls >= 1)
        & (falls < 100_000)
        & (jumps >= 1)
        & (jumps < 100_000)
        & (snap > 0)
        & (snap <= ticks)
        & (ticks - snap > 1000)
    )


class RunTimer:
    """Finds and reads the run-tick counter used by the wind."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.address: int | None = None
        self.last_value: int | None = None
        self._cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            address = int(payload["address"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        self.address = address

    def _save_cache(self) -> None:
        if self._cache_path is None or self.address is None:
            return
        try:
            self._cache_path.write_text(json.dumps({"address": self.address}), encoding="utf-8")
        except OSError:
            return

    def verify(self, pm: pymem.Pymem, window_s: float = 0.09) -> bool:
        """The cached heap address stays valid while the game process lives."""
        if self.address is None:
            return False
        try:
            before = struct.unpack("<i", pm.read_bytes(self.address, 4))[0]
            snapshot = struct.unpack("<i", pm.read_bytes(self.address - STATS_SIZE, 4))[0]
        except Exception:
            return False
        if before <= 0 or snapshot < 0 or snapshot > before:
            return False
        time.sleep(window_s)
        try:
            after = struct.unpack("<i", pm.read_bytes(self.address, 4))[0]
        except Exception:
            return False
        if 1 <= after - before <= 20:
            self.last_value = after
            return True
        return False

    def _scan(self, pm: pymem.Pymem) -> list[int]:
        hits: list[int] = []
        for base, size in iterate_regions(pm, private_only=True):
            for chunk_base, data in _read_chunks(pm, base, size):
                usable = len(data) - (len(data) % 4)
                if usable < 64:
                    continue
                arr = np.frombuffer(data[:usable], dtype=np.int32)
                if arr.size < 16:
                    continue
                ticks = arr[7:]
                prev = np.vstack([arr[6:-1], arr[5:-2], arr[4:-3], arr[3:-4], arr[2:-5]])
                snap = arr[0:-7]
                mask = _plausible_stats(prev, ticks, snap)
                for index in np.flatnonzero(mask).tolist():
                    hits.append(chunk_base + (index + 7) * 4)
        return hits

    def find(self, pm: pymem.Pymem, timeout_s: float = 15.0) -> bool:
        if self.verify(pm):
            self._save_cache()
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            candidates = self._scan(pm)[:MAX_TICK_CANDIDATES]
            first: dict[int, int] = {}
            for address in candidates:
                try:
                    value = struct.unpack("<i", pm.read_bytes(address, 4))[0]
                except Exception:
                    continue
                first[address] = value
            time.sleep(0.09)
            for address, before in first.items():
                try:
                    after = struct.unpack("<i", pm.read_bytes(address, 4))[0]
                except Exception:
                    continue
                if 1 <= after - before <= 20:
                    self.address = address
                    self.last_value = after
                    self._save_cache()
                    return True
        return False

    def ticks_delta(self, pm: pymem.Pymem) -> int | None:
        if self.address is None:
            return None
        try:
            all_ticks = struct.unpack("<i", pm.read_bytes(self.address, 4))[0]
            snapshot = struct.unpack("<i", pm.read_bytes(self.address - STATS_SIZE, 4))[0]
        except Exception:
            return None
        if all_ticks <= 0 or snapshot < 0:
            return None
        return all_ticks - snapshot

    def seconds(self, pm: pymem.Pymem) -> float | None:
        delta = self.ticks_delta(pm)
        if delta is None or delta < 0:
            return None
        return delta / 60.0
