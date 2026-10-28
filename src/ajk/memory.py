from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

import pymem

from .mems import scan_float_pairs


@dataclass
class PlayerMemory:
    process_name: str = "JumpKing.exe"
    address: int | None = None
    offset_x: float = 6.5
    offset_y: float = 26.0
    _pm: pymem.Pymem | None = field(default=None, repr=False)
    _failures: int = 0
    _pending: list[tuple[int, float, float]] = field(default_factory=list, repr=False)
    _pending_vision: tuple[float, float] = (0.0, 0.0)

    def attach(self) -> bool:
        try:
            self._pm = pymem.Pymem(self.process_name)
            return True
        except Exception:
            self._pm = None
            return False

    @property
    def available(self) -> bool:
        return self._pm is not None

    def rebase(self, d_row: int, d_col: int) -> None:
        if d_row == 0 and d_col == 0:
            return
        self.offset_y += d_row * 360.0
        self.offset_x -= d_col * 480.0

    def read_raw(self) -> tuple[float, float] | None:
        if self._pm is None or self.address is None:
            return None
        try:
            data = self._pm.read_bytes(self.address, 8)
            x, y = struct.unpack("<ff", data)
        except Exception:
            self._failures += 1
            return None
        if not (-1000.0 < x < 20000.0 and -1000.0 < y < 20000.0):
            self._failures += 1
            return None
        self._failures = 0
        return x, y

    def read(self) -> tuple[float, float] | None:
        raw = self.read_raw()
        if raw is None:
            return None
        return raw[0] + self.offset_x, raw[1] + self.offset_y

    def scan_pairs(
        self,
        near_x: float,
        near_y: float,
        tolerance: float = 40.0,
        max_results: int = 4000,
    ) -> list[tuple[int, float, float]]:
        if self._pm is None:
            return []
        return scan_float_pairs(
            self._pm,
            near_x - tolerance,
            near_x + tolerance,
            near_y - tolerance,
            near_y + tolerance,
            private_only=True,
            max_results=max_results,
        )

    def begin_calibration(self, vision_pos: tuple[float, float], tolerance: float = 60.0) -> int:
        raw_x = vision_pos[0] - self.offset_x
        raw_y = vision_pos[1] - self.offset_y
        self._pending = self.scan_pairs(raw_x, raw_y, tolerance=tolerance)
        self._pending_vision = vision_pos
        return len(self._pending)

    def finish_calibration(
        self, vision_pos: tuple[float, float], match_slack: float = 15.0
    ) -> bool:
        if not self._pending:
            return False
        dx = vision_pos[0] - self._pending_vision[0]
        dy = vision_pos[1] - self._pending_vision[1]
        if abs(dx) < 3 and abs(dy) < 3:
            return False
        best: tuple[float, int, float, float] | None = None
        expected_raw_x = self._pending_vision[0] - self.offset_x
        expected_raw_y = self._pending_vision[1] - self.offset_y
        for addr, x0, y0 in self._pending:
            try:
                x1, y1 = self._read_at(addr)
            except Exception:
                continue
            mdx = x1 - x0
            mdy = y1 - y0
            if abs(mdx) < 3 and abs(mdy) < 3:
                continue
            err = abs(mdx - dx) + abs(mdy - dy)
            if err > match_slack:
                continue
            prior = abs(x0 - expected_raw_x) + abs(y0 - expected_raw_y)
            score = err + prior * 0.35
            if best is None or score < best[0]:
                best = (score, addr, x1, y1)
        if best is None:
            return False
        self.address = best[1]
        self.offset_x = vision_pos[0] - best[2]
        self.offset_y = vision_pos[1] - best[3]
        return True

    def _read_at(self, addr: int) -> tuple[float, float]:
        data = self._pm.read_bytes(addr, 8)
        return struct.unpack("<ff", data)

    def wait_for_player(self, timeout_s: float = 20.0) -> tuple[float, float] | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            raw = self.read_raw()
            if raw is not None:
                return raw
            time.sleep(0.1)
        return None
