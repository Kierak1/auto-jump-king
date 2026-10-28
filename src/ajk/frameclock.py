"""Frame-locked input: hold keys for an exact number of game frames.

The game samples the keyboard once at the start of every 60 Hz frame and then
writes its `time_stamp` in PlayerEntity.Update. Observing that timestamp lets
us place key events inside a known frame interval, so a charge lasts exactly N
frames and the jump velocity is exact (v0 = JUMP_SPEED * N / 36).

Frame layout (one 60 Hz tick):

    input sampled --- BT runs --- entities update (time_stamp = S) --- draw
                                     ^ we observe S here
"""

from __future__ import annotations

import struct
import time

from .actions import Keyboard
from .savestate import PlayerMemoryV2

TIMESTAMP_OFFSET = 20


class FrameClock:
    def __init__(self, memory: PlayerMemoryV2) -> None:
        self.memory = memory
        self._last_stamp: int | None = None
        self._last_stamp_at = 0.0

    @property
    def address(self) -> int | None:
        return self.memory.address

    def stamp(self) -> int | None:
        pm = self.memory.pm
        address = self.memory.address
        if pm is None or address is None:
            return None
        try:
            data = pm.read_bytes(address + TIMESTAMP_OFFSET, 4)
        except Exception:
            return None
        value = struct.unpack("<i", data)[0]
        self._last_stamp = value
        self._last_stamp_at = time.perf_counter()
        return value

    def wait_stamp(
        self,
        target: int | None = None,
        timeout_s: float = 1.0,
        max_delta: int | None = None,
    ) -> int | None:
        """Wait until the frame counter changes (or reaches `target`).

        Returns the observed stamp, or None on timeout. `max_delta` rejects
        catch-up jumps (multiple updates in one draw) by returning None.
        """
        start = self.stamp()
        if start is None:
            return None
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            value = self.stamp()
            if value is None:
                return None
            if target is not None:
                if value >= target:
                    return value
                if max_delta is not None and value - start > max_delta:
                    return None
                continue
            if value != start:
                if max_delta is not None and value - start > max_delta:
                    return None
                return value
        return None

    def wait_ticks(self, ticks: int, timeout_s: float = 1.0) -> bool:
        """Wait for `ticks` frame boundaries; rejects catch-up multi-frames."""
        start = self.stamp()
        if start is None:
            return False
        return self.wait_stamp(start + ticks, timeout_s=timeout_s, max_delta=ticks + 1) is not None


class FrameKeyboard:
    """Keyboard input synchronised to the game's frame counter."""

    def __init__(self, keyboard: Keyboard, clock: FrameClock) -> None:
        self.keyboard = keyboard
        self.clock = clock

    def tap_frames(self, key: str, ticks: int = 1) -> bool:
        if not self.clock.wait_ticks(1):
            return False
        self.keyboard.down(key)
        if not self.clock.wait_ticks(ticks):
            self.keyboard.up(key)
            return False
        self.keyboard.up(key)
        return True

    def hold_until_landing(self, key: str, max_ticks: int = 120) -> None:
        start = self.clock.stamp()
        if start is None:
            self.keyboard.up(key)
            return
        self.clock.wait_stamp(start + max_ticks, timeout_s=max_ticks / 60.0 + 0.5)
        self.keyboard.up(key)

    def charge_jump(
        self,
        jump_key: str,
        direction_key: str | None,
        frames: int,
        timeout_s: float = 1.0,
    ) -> bool:
        """Jump with exactly `frames` charge frames (1..36, clamped).

        SPACE goes down first (aligned to a frame), then the direction key a
        couple of frames later while the charge is running, so the Walk node
        never moves the king. The game counts the charge in its own frames,
        so a hiccup in our polling (stamp jumps) does not change the charge.
        """
        frames = max(1, min(frames, 36))
        start = self.clock.stamp()
        if start is None:
            return False
        if direction_key is None:
            # JumpState.DoJump falls back to a short input buffer when no
            # direction is held at launch, so a straight-up jump would inherit
            # the walk direction from the last few frames; let the buffer
            # drain first (Buffer<State>(4) in the decompile).
            if not self.clock.wait_ticks(6):
                return False
            start = self.clock.stamp()
            if start is None:
                return False
        if not self.clock.wait_stamp(start + 1, timeout_s=timeout_s):
            return False
        press_stamp = self.clock.stamp()
        if press_stamp is None:
            return False
        self.keyboard.down(jump_key)
        if direction_key is not None:
            if frames <= 1:
                self.keyboard.down(direction_key)
            else:
                if not self.clock.wait_stamp(press_stamp + 2, timeout_s=timeout_s):
                    self.keyboard.up(jump_key)
                    return False
                self.keyboard.down(direction_key)
        if not self.clock.wait_stamp(press_stamp + frames, timeout_s=timeout_s + frames / 60.0):
            self.keyboard.up(jump_key)
            if direction_key is not None:
                self.keyboard.up(direction_key)
            return False
        self.keyboard.up(jump_key)
        if direction_key is not None:
            self.clock.wait_ticks(1)
            self.keyboard.up(direction_key)
        return True

    def walk_frames(self, key: str, frames: int) -> bool:
        return self.tap_frames(key, frames)

    def stop(self) -> None:
        self.keyboard.release_all()
