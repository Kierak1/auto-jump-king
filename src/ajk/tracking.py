"""Position tracking: memory/vision fusion, camera flips, ground snap.

Expects these attributes on the controller: cfg, logger, memory, memory_ok,
last_memory_pos, last_memory_attempt, tracker, screen_cell, screen_changed_at,
collision, state, history, view_offset, airborne, was_falling, stable_frames,
last_ground_y, detector, keyboard, verbose.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from .capture import GameView
from .physics import JumpPlan, TrajectoryPoint
from .state import PlayerState, TrackPoint
from .vision import CameraTracker, PlayerObs


class TrackingMixin:
    def _velocity(self) -> tuple[float, float]:
        if len(self.history) < 3:
            return 0.0, 0.0
        pts = list(self.history)[-6:]
        ts = np.array([p.t for p in pts])
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])
        ts = ts - ts[0]
        if ts[-1] - ts[0] < 1e-3:
            return 0.0, 0.0
        vx = float(np.polyfit(ts, xs, 1)[0])
        vy = float(np.polyfit(ts, ys, 1)[0])
        return vx, vy

    def _update_state(self, obs: PlayerObs | None, cam_x: float, cam_y: float) -> None:
        source = "vision"
        position: tuple[float, float] | None = None
        if self.memory_ok:
            memory_pos = self.memory.read()
            if memory_pos is not None:
                if self.last_memory_pos is not None:
                    jump = max(
                        abs(memory_pos[0] - self.last_memory_pos[0]),
                        abs(memory_pos[1] - self.last_memory_pos[1]),
                    )
                    if jump > 400.0:
                        memory_pos = None
                if memory_pos is not None:
                    position = memory_pos
                    source = "memory"
                    self.last_memory_pos = memory_pos
        clean_vision = obs is not None and not obs.merged
        if position is not None and source == "memory" and self.screen_cell is not None:
            flipped = self._check_screen_bounds(position)
            if flipped:
                new_pos = self.memory.read()
                if new_pos is not None:
                    position = new_pos
                    self.last_memory_pos = new_pos
        if position is None:
            if obs is None:
                return
            position = (obs.cx + cam_x, obs.bottom + cam_y)
            self.last_memory_pos = None
        elif clean_vision and self._now() - self.screen_changed_at > 0.5:
            vision_pos = (obs.cx + cam_x, obs.bottom + cam_y)
            correction_x = position[0] - vision_pos[0]
            correction_y = position[1] - vision_pos[1]
            drift = max(abs(correction_x), abs(correction_y))
            self.memory_vision_drift = drift
            if drift > 40.0:
                flipped = self._flip_from_drift(correction_x, correction_y)
                if flipped:
                    new_pos = self.memory.read()
                    if new_pos is not None:
                        position = new_pos
                        self.last_memory_pos = new_pos
                else:
                    self.memory_ok = False
                    self.memory.address = None
                    if self.verbose:
                        print(f"[memory] drift {drift:.0f}px, disabling memory tracking")
                    if self.logger:
                        self.logger.event("memory_drift", drift=drift)
                    position = vision_pos
                    self.last_memory_pos = None
        wx, wy = position
        if not self.airborne:
            snapped = self._snap_ground_y(wx, wy, source == "memory")
            if snapped is not None:
                wy = snapped
                if source == "memory":
                    self.last_memory_pos = (wx, wy)
        self.history.append(TrackPoint(self._now(), wx, wy))
        vx, vy = self._velocity()
        stable = False
        if len(self.history) >= 4:
            recent = [p.y for p in list(self.history)[-4:]]
            stable = max(recent) - min(recent) < 1.2
        self.stable_frames = self.stable_frames + 1 if stable else 0
        if vy < -30.0:
            self.airborne = True
            self.was_falling = False
        elif vy > 30.0:
            self.was_falling = True
        if self.airborne and self.was_falling and self.stable_frames >= 2:
            self.airborne = False
        grounded = (not self.airborne) and self.stable_frames >= 2
        self.state = PlayerState(
            x=wx,
            y=wy,
            vx=vx,
            vy=vy,
            grounded=grounded,
            screen_x=obs.cx if obs is not None else wx - cam_x,
            screen_y=obs.bottom if obs is not None else wy - cam_y,
            seen=True,
            source=source,
        )
        if grounded:
            if self.last_ground_y is None or self.last_ground_y < wy:
                self.last_ground_y = wy
            self.collision.mark_ground(wx, wy)

    def _check_screen_bounds(self, position: tuple[float, float]) -> bool:
        wx, wy = position
        d_row = d_col = 0
        if wy > 361.0:
            d_row = -1
        elif wy < -1.0:
            d_row = 1
        elif wx > 481.0:
            d_col = 1
        elif wx < -1.0:
            d_col = -1
        if d_row == 0 and d_col == 0:
            return False
        if not (-90.0 <= wx <= 570.0 and -90.0 <= wy <= 450.0):
            return False
        return self._apply_camera_flip(d_row, d_col)

    def _flip_from_drift(self, correction_x: float, correction_y: float) -> bool:
        flip_y = abs(abs(correction_y) - 360.0) < 60.0 and abs(correction_x) < 60.0
        flip_x = abs(abs(correction_x) - 480.0) < 60.0 and abs(correction_y) < 60.0
        if flip_y:
            return self._apply_camera_flip(-int(round(correction_y / 360.0)), 0)
        if flip_x:
            return self._apply_camera_flip(0, int(round(correction_x / 480.0)))
        return False

    def _apply_camera_flip(self, d_row: int, d_col: int) -> bool:
        if self.screen_cell is None:
            return False
        candidate = (self.screen_cell[0] + d_row, self.screen_cell[1] + d_col)
        if not (0 <= candidate[0] < 13 and 0 <= candidate[1] < 13):
            return False
        self.memory.rebase(d_row, d_col)
        if self.tracker is not None:
            self.tracker.realign(0.0, 0.0)
        self.view_offset = (0.0, 0.0)
        self.screen_changed_at = self._now()
        self.screen_cell = candidate
        self.collision.set_cell(candidate[0], candidate[1])
        self.blocked_takeoffs.clear()
        self.recent_failures.clear()
        shift_x = -d_col * 480.0
        shift_y = d_row * 360.0
        plans: list[JumpPlan] = []
        for holder in (self.current_plan, self.last_plan):
            if holder is not None:
                plans.append(holder)
        plans.extend(plan for _cand, plan in self.last_candidates)
        seen: set[int] = set()
        for plan in plans:
            if id(plan) in seen:
                continue
            seen.add(id(plan))
            plan.points = [
                TrajectoryPoint(p.t, p.x + shift_x, p.y + shift_y, p.vx, p.vy) for p in plan.points
            ]
            if plan.landing is not None:
                plan.landing = (plan.landing[0] + shift_x, plan.landing[1] + shift_y)
        if self.intended_landing is not None:
            self.intended_landing = (
                self.intended_landing[0] + shift_x,
                self.intended_landing[1] + shift_y,
            )
        self.history = deque(
            (TrackPoint(p.t, p.x + shift_x, p.y + shift_y) for p in self.history),
            maxlen=self.history.maxlen,
        )
        if self.verbose:
            print(f"[camera] flip d_row={d_row} d_col={d_col} cell={candidate}")
        if self.logger:
            self.logger.event("camera_flip", d_row=d_row, d_col=d_col, cell=list(candidate))
        return True

    def _atlas_surface_y(self, wx: float, wy: float, search: float = 24.0) -> float | None:
        if not (0.0 <= wx < 480.0):
            return None
        tx = int(wx // 8)
        start = max(0, int(wy // 8) - 1)
        end = int((wy + search) // 8)
        for ty in range(start, end + 1):
            if (tx, ty) in self.collision.tile_set:
                return float(ty * 8)
        return None

    def _snap_ground_y(self, wx: float, wy: float, adjust_memory: bool) -> float | None:
        if self.collision.atlas is None:
            return None
        surface = self._atlas_surface_y(wx, wy)
        if surface is None:
            return None
        delta = surface - wy
        if not (0.0 < abs(delta) <= 20.0):
            return None
        if adjust_memory:
            self.memory.offset_y += delta
        if self.verbose:
            print(f"[state] ground snap {delta:+.1f}px -> feet y={surface:.0f}")
        return surface

    def _memory_matches_atlas(self, pos: tuple[float, float]) -> bool:
        """Reject calibrations that locked onto a wrong object: the memory feet
        must sit on an atlas surface (the same test the ground snap uses)."""
        if self.collision.atlas is None or self.screen_cell is None:
            return True
        surface = self._atlas_surface_y(pos[0], pos[1])
        if surface is None:
            return False
        return abs(surface - pos[1]) <= 16.0

    def _setup_memory(self, view: GameView, tracker: CameraTracker) -> None:
        if not self.memory.attach():
            if self.verbose:
                print("[memory] not attached (process not found)")
            return
        deadline = self._now() + 3.0
        obs = None
        while self._now() < deadline:
            frame = view.grab()
            obs = self.detector.detect(frame, obs)
            if obs is not None and not obs.merged:
                break
            time.sleep(0.1)
        if obs is None or obs.merged:
            if self.verbose:
                print("[memory] player not cleanly visible, skipping calibration")
            return
        if self.screen_cell is None:
            self._update_screen(view.grab())
        if obs.cx > 240.0:
            keys = (self.cfg.input.left_key, self.cfg.input.right_key)
        else:
            keys = (self.cfg.input.right_key, self.cfg.input.left_key)
        for key in keys:
            before = (obs.cx, obs.bottom)
            count = self.memory.begin_calibration(before)
            if self.verbose:
                print(f"[memory] scanning {count} candidate positions ({key})...")
            if count == 0:
                continue
            self.keyboard.down(key)
            time.sleep(0.5)
            self.keyboard.up(key)
            time.sleep(0.2)
            frame = view.grab()
            after_obs = self.detector.detect(frame, obs)
            if after_obs is None:
                if self.verbose:
                    print("[memory] lost player during calibration")
                continue
            obs = after_obs
            if self.memory.finish_calibration((obs.cx, obs.bottom)):
                pos = self.memory.read()
                if pos is None or not self._memory_matches_atlas(pos):
                    if self.verbose:
                        print(f"[memory] rejected: {pos} does not sit on the atlas")
                    self.memory.address = None
                    continue
                opposite = (
                    self.cfg.input.left_key
                    if key == self.cfg.input.right_key
                    else self.cfg.input.right_key
                )
                self.keyboard.down(opposite)
                time.sleep(0.35)
                self.keyboard.up(opposite)
                time.sleep(0.2)
                verify_obs = self.detector.detect(view.grab(), obs)
                verify_pos = self.memory.read()
                if (
                    verify_obs is None
                    or verify_pos is None
                    or max(
                        abs(verify_pos[0] - verify_obs.cx),
                        abs(verify_pos[1] - verify_obs.bottom),
                    )
                    > 25.0
                ):
                    if self.verbose:
                        print("[memory] rejected: verification walk diverged")
                    self.memory.address = None
                    continue
                obs = verify_obs
                self.memory_ok = True
                if self.verbose:
                    print(
                        f"[memory] player position at {self.memory.address:#010x} "
                        f"offset=({self.memory.offset_x:+.1f},{self.memory.offset_y:+.1f})"
                    )
                if self.logger:
                    self.logger.event(
                        "memory_calibrated",
                        address=self.memory.address,
                        offset=[self.memory.offset_x, self.memory.offset_y],
                    )
                return
        if self.verbose:
            print("[memory] calibration failed, using vision only")
