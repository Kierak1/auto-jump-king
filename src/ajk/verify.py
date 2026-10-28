"""Verify the exact simulator against the live game (frame-locked jumps)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mss
import numpy as np

from .actions import Keyboard
from .frameclock import FrameClock, FrameKeyboard
from .gamephysics import JUMP_SPEED, WorldGrid, simulate_jump
from .savestate import PlayerMemoryV2, PlayerSaveState, read_save_state
from .vision import PlayerDetector
from .win import WindowInfo, focus, refresh


def vision_hint(window: WindowInfo) -> tuple[float, float] | None:
    focus(window.hwnd)
    time.sleep(0.2)
    info = refresh(window)
    with mss.mss() as sct:
        raw = np.asarray(
            sct.grab(
                {
                    "left": info.client_x,
                    "top": info.client_y,
                    "width": info.client_w,
                    "height": info.client_h,
                }
            ),
            dtype=np.uint8,
        )[:, :, :3]
    frame = cv2.resize(raw, (480, 360))
    obs = PlayerDetector().detect(frame)
    if obs is None:
        return None
    return obs.cx - 9.0, obs.bottom - 26.0


@dataclass
class Trial:
    request: int
    direction: int
    start: tuple[float, float]
    charge: float = 0.0
    trace: list[tuple[float, float, float, float]] = field(default_factory=list)
    landing: tuple[float, float] | None = None
    sim_landing: tuple[float, float] | None = None


class StateSampler:
    def __init__(self, memory: PlayerMemoryV2) -> None:
        self.memory = memory
        self.samples: list[PlayerSaveState] = []
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last = None
        while not self._stop:
            state = read_save_state(self.memory.pm, self.memory.address)
            if state is not None and state.time_stamp != last:
                last = state.time_stamp
                self.samples.append(state)
            time.sleep(0.0002)

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)


def run_trial(
    memory: PlayerMemoryV2,
    grid: WorldGrid,
    clock: FrameClock,
    keyboard: Keyboard,
    frame_keyboard: FrameKeyboard,
    frames: int,
    direction: int,
    window: WindowInfo,
) -> Trial | None:
    if not memory.ensure_address(*(vision_hint(window) or (None, None))):
        return None
    state = read_save_state(memory.pm, memory.address)
    if state is None or not state.grounded:
        return None
    trial = Trial(request=frames, direction=direction, start=(state.x, state.y))
    sampler = StateSampler(memory)
    sampler.start()
    time.sleep(0.1)
    key = {1: "RIGHT", -1: "LEFT"}.get(direction)
    ok = frame_keyboard.charge_jump("SPACE", key, frames)
    time.sleep(1.2)
    sampler.stop()
    if not ok or not sampler.samples:
        return None
    airborne = False
    takeoff: PlayerSaveState | None = None
    for sample in sampler.samples:
        trial.trace.append((sample.x, sample.y, sample.vx, sample.vy))
        if takeoff is None:
            if sample.vy < -0.5:
                takeoff = sample
                trial.charge = -sample.vy / JUMP_SPEED * 36.0
            continue
        if not sample.grounded:
            airborne = True
        elif airborne and sample.vy >= 0:
            trial.landing = (sample.x, sample.y + 26.0)
            break
    if takeoff is None:
        return None
    trial.start = (takeoff.x, takeoff.y)
    sim = simulate_jump(grid, takeoff.x, takeoff.y, direction, frames)
    trial.sim_landing = sim.landing
    return trial


def print_trial(trial: Trial) -> None:
    print(
        f"request={trial.request} dir={trial.direction:+d} "
        f"charge={trial.charge:.2f} "
        f"actual={trial.landing} sim={trial.sim_landing}"
    )


def run_verification(
    window: WindowInfo,
    game_dir: Path,
    requests: list[int],
    directions: list[int],
    trials_per: int = 1,
) -> int:
    memory = PlayerMemoryV2()
    memory.attach()
    grid = WorldGrid(game_dir)
    clock = FrameClock(memory)
    keyboard = Keyboard()
    frame_keyboard = FrameKeyboard(keyboard, clock)
    mismatches = 0
    total = 0
    for _ in range(trials_per):
        for direction in directions:
            for frames in requests:
                trial = run_trial(
                    memory, grid, clock, keyboard, frame_keyboard, frames, direction, window
                )
                if trial is None:
                    print(f"request={frames} dir={direction:+d}: skipped (not grounded)")
                    time.sleep(0.3)
                    continue
                print_trial(trial)
                total += 1
                if trial.landing is None or trial.sim_landing is None:
                    mismatches += 1
                else:
                    error = max(
                        abs(trial.landing[0] - trial.sim_landing[0]),
                        abs(trial.landing[1] - trial.sim_landing[1]),
                    )
                    if error > 2.0:
                        mismatches += 1
                time.sleep(0.2)
    print(f"{total - mismatches}/{total} jumps matched the simulation")
    return 0 if mismatches == 0 else 1
