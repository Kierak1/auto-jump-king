"""Frame-exact autonomous player driven by the backward chain planner.

State comes from the game's own SaveState struct in memory (exact position,
velocity, grounded, frame counter). The planner takes the platforms above the
king and searches BACKWARD from the highest one for a chain of verified jumps
that connects to his feet (see chains.py). Every jump is planned with the exact
physics simulator and executed with frame-locked keyboard timing, then verified
against the planned landing.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

from .actions import Keyboard, key_is_down
from .chains import WINDOW_PLATFORMS, Chain, ChainPlanner
from .edges import MIN_MARGIN, WIND_EPOCH_BASE, Edge, EdgeFactory
from .frameclock import FrameClock, FrameKeyboard
from .gamephysics import (
    JUMP_SPEED,
    PLAYER_HEIGHT,
    WALK_SPEED,
    WIND_OMEGA,
    WIND_STRENGTH,
    WorldGrid,
    charge_intensity,
    wind_velocity,
)
from .planner import Option, extract_platforms, options_for, platform_below, surface_run
from .runtimer import RunTimer
from .savestate import PlayerMemoryV2, PlayerSaveState, read_save_state
from .simulate import ICE_FRICTION, JumpResult, StepEnv, simulate_fall, simulate_jump
from .telemetry import RunLogger
from .verify import vision_hint
from .win import WindowInfo, focus, is_foreground

GOAL_SCREEN = 42
GOAL_CENTER_X = 240.0
STUCK_SECONDS = 20.0
LANDING_TOLERANCE = 4.0
SPLAT_STATE = 2
SPLAT_TAP_SECONDS = 0.5
PLAN_SECONDS = 12.0
PLAN_SECONDS_WIND = 4.0
WINDOW_PLATFORMS_WIND = 14
WIND_FIT_SAMPLES = 60
WIND_FIT_RMS = 0.07
WIND_REFIT_RMS = 0.015
WIND_REFIT_RADIUS_S = 2.0
WIND_REFINE_RADIUS_S = 0.5
WIND_LAUNCH_LEAD = 1.0
# Wind edges are cached in phase buckets within the wind period: every plan
# then reuses edges from the same bucket instead of recomputing the whole
# window at a unique (and always cold) absolute epoch. The buckets are wide so
# consecutive decisions share one; execution verifies the exact phase anyway.
WIND_EPOCH_SECONDS = 3.0
SNOW_HOP_FRAMES = 11
SNOW_HOP_LOOKAHEAD = 3
SNOW_HOP_LEAD = 0.4
SNOW_HOP_SECONDS = 1.3
SNOW_ARC_LEAD = 0.6
SNOW_WAIT_SECONDS = 1.0


@dataclass
class RunStats:
    jumps: int = 0
    falls: int = 0
    decisions: int = 0
    misses: int = 0
    best_y: float = 1e9
    best_screen: int = 0
    started_at: float = 0.0
    progress_at: float = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "jumps": self.jumps,
            "falls": self.falls,
            "decisions": self.decisions,
            "misses": self.misses,
            "best_y": round(self.best_y, 1),
            "best_screen": self.best_screen,
            "elapsed_s": round(time.monotonic() - self.started_at, 1),
        }


class ExactController:
    def __init__(self, cfg, window: WindowInfo, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.window = window
        self.logger = logger
        self.keyboard = Keyboard(tap_ms=cfg.input.tap_ms)
        self.memory = PlayerMemoryV2()
        self.clock = FrameClock(self.memory)
        self.frame_keyboard = FrameKeyboard(self.keyboard, self.clock)
        self.grid = WorldGrid(Path(cfg.game_dir))
        self.platforms = extract_platforms(self.grid, min_width=8.0)
        self.edge_factory = EdgeFactory(
            self.grid,
            self.platforms,
            cache_path=Path(cfg.play.runs_dir) / "edge_cache.pkl",
        )
        self.chain_planner = ChainPlanner(self.grid, self.platforms, self.edge_factory)
        self.stats = RunStats()
        self.state: PlayerSaveState | None = None
        self.screen = 0
        self.phase = "idle"
        self.chain: Chain | None = None
        self.edge: Edge | None = None
        self.walk_target = 0.0
        self.walk_last_x: float | None = None
        self.walk_last_move = 0.0
        self.planned_landing: tuple[float, float] | None = None
        self.planned_arc: list[tuple[float, float]] = []
        self.takeoff_pos: tuple[float, float] | None = None
        self.takeoff_stamp = 0
        self.expected_vy: float | None = None
        self.bad_liftoff = False
        self.blacklist: set[tuple[float, int, int]] = set()
        self.blacklist_until = 0.0
        self.cooldown_until = 0.0
        self.last_landing_error: tuple[float, float] | None = None
        self.last_options: list[Option] = []
        self.last_chosen: Option | None = None
        self.ending_detected = False
        self.verbose = False
        self.current_frame = None
        self._last_screen_dump = -1
        self._dump_ui_label: str | None = None
        self._frame_counter = 0
        self.run_timer = RunTimer(cache_path=Path(cfg.play.runs_dir) / "run_timer.json")
        self.wind_offset: float | None = self._load_wind_phase()
        self.wind_validated = False
        self._wind_calibrating = False
        self.wind_enabled = False
        self.wind_deltas: list[tuple[float, float]] = []
        self._stand_sample: tuple[float, int] | None = None
        self.wind_gate = False
        self.wind_fit_count = 0
        self.last_plan_seconds = WIND_LAUNCH_LEAD
        self.takeoff_wait_until = 0.0
        self.recent_platforms: deque[int] = deque(maxlen=6)
        self.visit_counts: dict[int, int] = {}
        self.banned_platforms: set[int] = set()
        self.flight_trace: list[tuple] = []
        self._menu_hits = 0
        self._plan_thread: threading.Thread | None = None
        self._plan_result: Chain | None = None
        self._recover_thread: threading.Thread | None = None
        self._recover_result: list[Option] | None = None
        self.plan_hold_x: float | None = None
        self.splat_tap_at = 0.0
        self.deadline = float("inf")

    def log(self, kind: str, **data: object) -> None:
        if self.logger is not None:
            self.logger.event(kind, **data)

    def _read(self) -> PlayerSaveState | None:
        state = read_save_state(self.memory.pm, self.memory.address)
        if state is not None and state.x == 0.0 and state.y == 0.0:
            state = None
        if state is None:
            if self.memory.address is not None:
                self.memory.address = None
            hint = vision_hint(self.window)
            self.memory.ensure_address(*(hint or (None, None)), timeout_s=10.0)
            state = read_save_state(self.memory.pm, self.memory.address)
            if state is not None and state.x == 0.0 and state.y == 0.0:
                return None
        return state

    def _ensure_focus(self) -> None:
        if not is_foreground(self.window.hwnd):
            focus(self.window.hwnd)

    # ------------------------------------------------------------------ wind

    def _screen_of(self, y: float) -> int:
        return -int((y + 13.0) // 360.0)

    def _in_wind_region(self, state: PlayerSaveState) -> bool:
        return self._screen_of(state.y) in self.grid.wind_screens

    def _update_wind_enabled(self, state: PlayerSaveState) -> bool:
        screen = self._screen_of(state.y)
        if screen not in self.grid.wind_screens:
            return False
        if state.grounded:
            return True
        return self.wind_enabled

    def _harvest_wind_deltas(self, state: PlayerSaveState) -> None:
        """WindManager.CurrentVelocityRaw adds itself to velocity.X on every
        airborne frame (BodyComp.UpdateWind, decompiled), so consecutive
        one-frame samples of the flight trace expose the raw wind per frame.
        The p_wind gate is the SCREEN flag (green tiles mark the screen) minus
        white no-wind tiles, per LevelScreen.TryCollision."""
        trace = self.flight_trace
        if len(trace) < 2:
            return
        if not self.wind_gate:
            # The game only enables the wind on ground contact in a wind
            # screen (or when falling down into one); a flight that entered
            # the screen from below sees no wind at all.
            return
        pm = self.memory.pm
        timer_seconds = None
        if self.run_timer.address is not None and pm is not None:
            timer_seconds = self.run_timer.seconds(pm)
        # Fitting against the exact run timer is stable; the SaveState frame
        # counter drifts relative to it across player entities, so without the
        # timer the samples are not used for a fit at all.
        if timer_seconds is None:
            return
        for (stamp0, x0, y0, vx0, _vy0), (stamp1, _x1, _y1, vx1, _vy1) in zip(
            trace, trace[1:], strict=False
        ):
            # The control loop samples every couple of frames; the wind is
            # essentially constant over that span (period ~13s), so the mean
            # per-frame delta is a clean sample.
            dt = stamp1 - stamp0
            if not 1 <= dt <= 4:
                continue
            if self._screen_of(y0) not in self.grid.wind_screens:
                continue
            if self.grid.has_nowind(x0, y0):
                continue
            delta = vx1 - vx0
            if abs(delta) > 0.12 * dt + 0.02:
                continue
            # Sample coordinate is the run-timer clock, the same clock the
            # game derives the wind phase from.
            sample_seconds = timer_seconds - (state.time_stamp - stamp0) / 60.0
            self._record_wind_sample(sample_seconds, delta / dt)

    def _record_wind_sample(self, seconds: float, wind: float) -> None:
        self.wind_deltas.append((seconds, wind))
        if len(self.wind_deltas) > 600:
            del self.wind_deltas[: len(self.wind_deltas) - 600]
        self._maybe_fit_wind()

    def _sample_standing_wind(self, state: PlayerSaveState) -> None:
        """While the king stands still in a wind screen the wind slides him by
        one wind step per frame (`Walk` zeroes velocity.X every frame and the
        wind adds it back), so consecutive positions expose the wind directly.
        This gives dense samples without waiting for airborne flights."""
        if (
            not state.grounded
            or self.phase not in ("idle", "takeoff")
            or self._screen_of(state.y) not in self.grid.wind_screens
            or self._on_snow(state.x, state.y)
            or self.grid.has_nowind(state.x, state.y)
        ):
            self._stand_sample = None
            return
        from .slopes import collision_info

        if collision_info(self.grid, state.x, state.y + PLAYER_HEIGHT + 1.0).slope != 0:
            # Slopes slide the king; that motion is not wind.
            self._stand_sample = None
            return
        previous = self._stand_sample
        self._stand_sample = (state.x, state.time_stamp)
        if previous is None:
            return
        dt = state.time_stamp - previous[1]
        if not 1 <= dt <= 4:
            return
        wind = (state.x - previous[0]) / dt
        if abs(wind) > 0.12:
            return
        pm = self.memory.pm
        if self.run_timer.address is None or pm is None:
            return
        seconds = self.run_timer.seconds(pm)
        if seconds is None:
            return
        self._record_wind_sample(seconds - dt / 60.0, wind)

    def _maybe_fit_wind(self) -> None:
        fresh = len(self.wind_deltas) - self.wind_fit_count
        if fresh < WIND_FIT_SAMPLES:
            return
        if self.wind_offset is not None:
            # Validate the persisted/current phase against the fresh samples;
            # only move when the current offset clearly fails to explain them.
            current_rms = self._wind_rms(self.wind_deltas, self.wind_offset)
            if current_rms <= WIND_REFIT_RMS:
                self.wind_fit_count = len(self.wind_deltas)
                self.wind_validated = True
                return
            fit = self._fit_wind_phase(self.wind_deltas, self.wind_offset, WIND_REFIT_RADIUS_S)
            if fit is None or fit[1] > WIND_FIT_RMS:
                # The near search failed; only then allow a global refit (the
                # periodic wind aliases to far minima that can otherwise win).
                fit = self._fit_wind_phase(self.wind_deltas)
            if fit is not None and fit[1] <= WIND_FIT_RMS:
                self.wind_fit_count = len(self.wind_deltas)
                self.log(
                    "wind_refit",
                    old=round(self.wind_offset, 2),
                    new=round(fit[0], 2),
                    old_rms=round(current_rms, 4),
                    rms=round(fit[1], 4),
                )
                self.wind_offset = fit[0]
                self.wind_validated = True
                self._save_wind_phase()
            else:
                self.wind_fit_count = len(self.wind_deltas)
                self.log("wind_phase_stale", old_rms=round(current_rms, 4))
                self.wind_offset = None
                self.wind_validated = False
            return
        fit = self._fit_wind_phase(self.wind_deltas)
        if fit is None:
            return
        offset, rms = fit
        self.wind_fit_count = len(self.wind_deltas)
        if rms > WIND_FIT_RMS:
            self.log("wind_fit_failed", rms=round(rms, 4), samples=len(self.wind_deltas))
            self._dump_wind_samples(rms)
            return
        self.wind_offset = offset
        self.wind_validated = True
        self.log(
            "wind_fit",
            offset=round(offset, 2),
            rms=round(rms, 4),
            samples=len(self.wind_deltas),
        )
        self._save_wind_phase()

    def _dump_wind_samples(self, rms: float) -> None:
        """Diagnostic: persist the sample set that failed to fit."""
        try:
            path = Path(self.cfg.play.runs_dir) / "wind_samples_debug.json"
            path.write_text(json.dumps({"rms": rms, "samples": self.wind_deltas}), encoding="utf-8")
        except OSError:
            return

    def _wind_phase_path(self) -> Path:
        return Path(self.cfg.play.runs_dir) / "wind_phase.json"

    def _load_wind_phase(self) -> float | None:
        """The phase offset between the run timer and the wind function is a
        constant, so it survives across runs (domain-tagged to avoid mixing it
        with the older SaveState-frame fits)."""
        try:
            payload = json.loads(self._wind_phase_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if payload.get("domain") != "timer_seconds2":
            return None
        offset = payload.get("offset")
        return float(offset) if isinstance(offset, (int, float)) else None

    def _save_wind_phase(self) -> None:
        if self.wind_offset is None:
            return
        try:
            self._wind_phase_path().write_text(
                json.dumps({"domain": "timer_seconds2", "offset": round(self.wind_offset, 3)}),
                encoding="utf-8",
            )
        except OSError:
            return

    def _fit_wind_phase(
        self,
        samples: list[tuple[float, float]],
        center: float | None = None,
        radius_s: float | None = None,
    ) -> tuple[float, float] | None:
        """Align `wind_velocity(run_seconds + offset)` with the observed
        per-frame wind. The wind flips sign abruptly at the model's
        discontinuity, so a couple of frames of phase error produce huge
        residuals there; the fit is therefore refined with the worst quarter
        of the samples trimmed. Returns (offset_seconds, rms) or None.

        When `center` is given the search stays within `radius_s` of it, so
        the periodic function cannot alias to a half-period-away minimum once
        a phase is already known.
        """
        if not samples:
            return None
        offset = self._best_wind_offset(samples, center, radius_s)
        trimmed = self._trim_wind_outliers(samples, offset, fraction=0.25)
        offset = self._best_wind_offset(trimmed, offset, WIND_REFINE_RADIUS_S)
        residuals = sorted((wind_velocity(stamp + offset) - wind) ** 2 for stamp, wind in trimmed)
        keep = max(1, int(len(residuals) * 0.75))
        rms = math.sqrt(sum(residuals[:keep]) / keep)
        return offset, rms

    @staticmethod
    def _best_wind_offset(
        samples: list[tuple[float, float]],
        center: float | None = None,
        radius_s: float | None = None,
    ) -> float:
        """Two-stage phase search: a coarse scan over the span, then a fine
        refinement around the best candidate (the period is smooth apart from
        the sign flips, which the sample set pins down)."""
        period_s = 2.0 * math.pi / WIND_OMEGA
        if center is None:
            base, span = 0.0, period_s
        else:
            span = radius_s if radius_s is not None else 0.0
            base = center - span
        coarse = 1.0 / 15.0

        def error_at(offset: float) -> float:
            total = 0.0
            for stamp, wind in samples:
                total += (wind_velocity(stamp + offset) - wind) ** 2
            return total

        best_offset = base
        best_error = float("inf")
        offset = base
        limit = base + 2.0 * span
        while offset <= limit:
            value = error_at(offset)
            if value < best_error:
                best_error = value
                best_offset = offset
            offset += coarse
        fine = 1.0 / 300.0
        offset = best_offset - coarse
        limit = best_offset + coarse
        while offset <= limit:
            value = error_at(offset)
            if value < best_error:
                best_error = value
                best_offset = offset
            offset += fine
        return best_offset

    @staticmethod
    def _wind_rms(samples: list[tuple[float, float]], offset: float) -> float:
        if not samples:
            return float("inf")
        error = 0.0
        for stamp, wind in samples:
            error += (wind_velocity(stamp + offset) - wind) ** 2
        return math.sqrt(error / len(samples))

    @staticmethod
    def _trim_wind_outliers(
        samples: list[tuple[float, float]], offset: float, fraction: float
    ) -> list[tuple[float, float]]:
        if not samples:
            return samples
        scored = sorted(
            samples,
            key=lambda item: abs(wind_velocity(item[0] + offset) - item[1]),
        )
        keep = max(1, int(len(scored) * (1.0 - fraction)))
        return scored[:keep]

    def _wind_start(self, lead: float = 0.0) -> float | None:
        """Run time in seconds for the wind model, from the game's own attempt
        timer (AchievementManager.GetCurrentStats). The fitted `wind_offset`
        aligns `wind_velocity` with the timer's phase."""
        pm = self.memory.pm
        if self.run_timer.address is None or pm is None:
            return None
        seconds = self.run_timer.seconds(pm)
        if seconds is None:
            return None
        return seconds + (self.wind_offset or 0.0) + lead

    def _wind_now(self) -> float:
        if self.wind_offset is None:
            return 0.0
        start = self._wind_start()
        if start is None:
            return 0.0
        return wind_velocity(start)

    def _wind_risk(self, state: PlayerSaveState) -> bool:
        """True when a planning pause would let the wind slide the king off
        the platform (wind screens only, snow blocks the wind)."""
        if self.wind_offset is None or not self.wind_enabled:
            return False
        if self._screen_of(state.y) not in self.grid.wind_screens:
            return False
        if self._on_snow(state.x, state.y) or self.grid.has_nowind(state.x, state.y):
            return False
        wind = self._wind_now()
        if abs(wind) < 0.02:
            return False
        run_x0, run_x1 = surface_run(self.grid, state.x + 9.0, state.y)
        margin = 28.0
        if wind > 0.0:
            return (run_x1 - 1.0) - state.x < margin
        return state.x - (run_x0 - 17.0) < margin

    def _wind_anchor(self, state: PlayerSaveState) -> float:
        """A parking spot on the upwind quarter of the current run, far enough
        from the downwind edge to survive a slow plan."""
        wind = self._wind_now()
        run_x0, run_x1 = surface_run(self.grid, state.x + 9.0, state.y)
        supported_left = run_x0 - 17.0
        supported_right = run_x1 - 1.0
        span = max(8.0, supported_right - supported_left)
        offset = min(24.0, span * 0.25)
        if wind > 0.0:
            return min(supported_left + offset, supported_right - 4.0)
        return max(supported_right - offset, supported_left + 4.0)

    def _configure_wind(self, state: PlayerSaveState) -> None:
        self.wind_enabled = self._update_wind_enabled(state)
        if not self.wind_enabled:
            self.edge_factory.set_wind(None, False, 0)
            return
        start = self._wind_start()
        if start is None:
            self.edge_factory.set_wind(None, True, 0)
            return
        env = StepEnv(wind=wind_velocity, start_time=start)
        epoch = WIND_EPOCH_BASE + int((start % (2.0 * math.pi / WIND_OMEGA)) / WIND_EPOCH_SECONDS)
        self.edge_factory.set_wind(env, True, epoch)

    def _configure_wind_for_launch(self, x: float, y: float) -> None:
        """Plan with the wind phase expected at the actual launch: the last
        plan's duration plus the charge elapse before the jump."""
        if not self.wind_enabled:
            return
        lead = WIND_LAUNCH_LEAD + min(self.last_plan_seconds, 1.5)
        start = self._wind_start(lead)
        if start is None:
            return
        env = StepEnv(wind=wind_velocity, start_time=start)
        epoch = WIND_EPOCH_BASE + int((start % (2.0 * math.pi / WIND_OMEGA)) / WIND_EPOCH_SECONDS)
        self.edge_factory.set_wind(env, True, epoch)

    # ------------------------------------------------------------- planning

    def _plan_avoid(self, state: PlayerSaveState) -> set[int]:
        """Visit bookkeeping plus the banned-platform set for one plan."""
        self.chain_planner.blocked = self.blacklist
        current = platform_below(self.platforms, state.x, state.y + PLAYER_HEIGHT)
        avoid = set(self.recent_platforms)
        if current is not None:
            index = self.chain_planner.index_of[id(current)]
            if not self.recent_platforms or self.recent_platforms[-1] != index:
                self.recent_platforms.append(index)
                self.visit_counts[index] = self.visit_counts.get(index, 0) + 1
                if (
                    self.visit_counts[index] >= 4
                    and time.monotonic() - self.stats.progress_at > 6.0
                ):
                    self.banned_platforms.add(index)
                    self.log(
                        "ban_platform",
                        top_y=round(current.top_y, 1),
                        x0=round(current.x0, 1),
                        x1=round(current.x1, 1),
                        visits=self.visit_counts[index],
                    )
            avoid = set(self.recent_platforms)
            avoid.discard(index)
        return avoid | self.banned_platforms

    def _plan_worker(self, x: float, y: float, avoid: set[int]) -> None:
        try:
            self._plan_result = self._bounded_plan(x, y, avoid)
        except Exception as exc:  # keep the worker from dying silently
            self._plan_result = None
            self.log("plan_error", error=repr(exc))

    def _plan_hold(self, state: PlayerSaveState) -> None:
        """Keep the king near the position the running plan was made from:
        the wind slides him while the worker plans, so tap against the drift."""
        left = self.cfg.input.left_key
        right = self.cfg.input.right_key
        if not state.grounded or self.plan_hold_x is None:
            self.keyboard.up(left)
            self.keyboard.up(right)
            return
        error = self.plan_hold_x - state.x
        if abs(error) <= 0.75:
            self.keyboard.up(left)
            self.keyboard.up(right)
            return
        key = right if error > 0 else left
        other = left if key == right else right
        self.keyboard.up(other)
        self.frame_keyboard.tap_frames(key, 1)

    def _bounded_plan(self, x: float, y: float, avoid: set[int]) -> Chain | None:
        """Run one plan with a per-decision time cap so a cold area can never
        stall the control loop past the run deadline. Completed edge pairs are
        cached, so repeated attempts make incremental progress. Wind screens
        use a shorter horizon and cap: the king drifts while we plan and long
        chains are unreliable in wind anyway. Runs on the planning worker
        while the control loop keeps holding the king in place."""
        wind = self.wind_offset is not None and self.wind_enabled
        window_size = WINDOW_PLATFORMS_WIND if wind else WINDOW_PLATFORMS
        cap = PLAN_SECONDS_WIND if wind else PLAN_SECONDS
        self.edge_factory.deadline = min(self.deadline, time.monotonic() + cap)
        self._configure_wind_for_launch(x, y)
        started = time.monotonic()
        try:
            return self.chain_planner.plan(x, y, window_size, avoid=avoid)
        finally:
            self.last_plan_seconds = time.monotonic() - started
            self.edge_factory.deadline = self.deadline

    def _wind_ready(self, state: PlayerSaveState) -> bool:
        """False while the wind phase is not yet validated in this process.

        A persisted offset belongs to a timer baseline that may have changed
        (new game process or attempt), and a wrong offset predicts the wind
        backwards; the bot idles until fresh samples confirm or replace it.
        """
        if self.run_timer.address is None or self.wind_validated:
            return True
        if self._screen_of(state.y) not in self.grid.wind_screens or not state.grounded:
            return True
        if self._on_snow(state.x, state.y) or self.grid.has_nowind(state.x, state.y):
            return True
        from .slopes import collision_info

        return collision_info(self.grid, state.x, state.y + PLAYER_HEIGHT + 1.0).slope != 0

    def _decide(self, state: PlayerSaveState) -> None:
        if not self._wind_ready(state):
            if not self._wind_calibrating:
                self._wind_calibrating = True
                self.log("wind_calibrating", samples=len(self.wind_deltas))
            return
        self._wind_calibrating = False
        if self._wind_risk(state):
            # In a wind screen the king drifts while we plan; near the
            # downwind edge a slow plan slides him off, so park upwind first.
            anchor = self._wind_anchor(state)
            if abs(anchor - state.x) > 3.0:
                self.log("wind_hold", x=round(state.x, 1), anchor=round(anchor, 1))
                self._start_walk(anchor)
                return
        thread = self._plan_thread
        if thread is not None and thread.is_alive():
            # Keep the king in place while the worker computes the chain.
            self._plan_hold(state)
            return
        if thread is not None:
            thread.join(timeout=0.0)
            self._plan_thread = None
            chain = self._plan_result
            self._plan_result = None
            self._plan_hold_x = None
            if (chain is None or chain.first is None) and self.banned_platforms:
                self.log("bans_cleared", banned=len(self.banned_platforms))
                self.banned_platforms.clear()
                self.visit_counts.clear()
                self._start_plan(state, set(self.recent_platforms))
                return
            self._execute_chain(state, chain)
            return
        rthread = self._recover_thread
        if rthread is not None:
            if rthread.is_alive():
                self._plan_hold(state)
                return
            rthread.join(timeout=0.0)
            self._recover_thread = None
            options = self._recover_result or []
            self._recover_result = None
            self._finish_recover(state, options)
            return
        self._start_plan(state, self._plan_avoid(state))

    def _start_plan(self, state: PlayerSaveState, avoid: set[int]) -> None:
        self.plan_hold_x = state.x
        self._plan_result = None
        self._plan_thread = threading.Thread(
            target=self._plan_worker, args=(state.x, state.y, avoid), daemon=True
        )
        self._plan_thread.start()
        self.log("plan_start", x=round(state.x, 1), y=round(state.y, 1))

    def _execute_chain(self, state: PlayerSaveState, chain: Chain | None) -> None:
        self.last_options = []
        if chain is None or chain.first is None:
            self.chain = None
            self._recover(state)
            return
        # A long chain may legitimately pass back through a recently visited
        # platform (e.g. the drop that opens the screen-5 route); only short
        # ping-pong chains are worth abandoning, and persistent loops are
        # handled by the visit-count bans.
        if (
            chain.revisit
            and len(chain.links) <= 2
            and time.monotonic() - self.stats.progress_at > 6.0
        ):
            self.chain = None
            self.log("revisit_recover", x=round(state.x, 1), y=round(state.y, 1))
            self._recover(state)
            return
        self.chain = chain
        self.stats.decisions += 1
        edge = chain.first.edge
        self.edge = edge
        if self._on_snow(state.x, state.y):
            edge = self._snow_decision(state, edge)
            if edge is None:
                return
        self.edge = edge
        self.last_chosen = None
        self.takeoff_wait_until = 0.0
        self.log(
            "decision",
            links=len(chain.links),
            robust=chain.robust,
            choice=edge.describe(),
            takeoff=round(edge.takeoff_x, 1),
            x=round(state.x, 1),
            target=round(self.platforms[edge.target].top_y, 1),
            tried=chain.tried,
        )
        self._takeoff(state)

    def plan_idle(self) -> bool:
        """True while no planning or recovery worker is running."""
        for thread in (self._plan_thread, self._recover_thread):
            if thread is not None and thread.is_alive():
                return False
        return True

    # ------------------------------------------------------------- movement

    def _start_walk(self, target: float) -> None:
        self.phase = "walk"
        self.walk_target = target
        self.walk_last_x = None
        self.walk_last_move = time.monotonic()

    def _on_snow(self, x: float, y: float) -> bool:
        from .slopes import collision_info

        # Sample the whole footprint just under the feet: at a platform edge
        # the hitbox can hang over empty tiles while still standing on snow.
        return collision_info(self.grid, x, y + PLAYER_HEIGHT + 1.0).snow

    def _snow_decision(self, state: PlayerSaveState, edge: Edge) -> Edge | None:
        """Pick the executable edge while standing on snow.

        Snow zeroes the walk velocity, so a planned takeoff can only be reached
        by jumping. If an edge to the target already lands from where the king
        stands, execute it from here (the exact wind-aware simulation is the
        proof). Otherwise search the short graph of minimum snow hops for the
        fewest hops that reach a position where the planned arc verifies at the
        wind phase that hop sequence would actually launch in; if none exists,
        wait for the phase to move instead of blacklisting into a walk-off.
        """
        current = platform_below(self.platforms, state.x, state.y + PLAYER_HEIGHT)
        index = self.chain_planner.index_of[id(current)] if current is not None else None
        candidates = [edge]
        if index is not None:
            candidates.extend(self.edge_factory.edges_to(edge.target, index))
        for candidate in candidates:
            result = self._verified_at(candidate, self._launch_x(state), state.y, SNOW_ARC_LEAD)
            if result is not None and result.landing is not None:
                return replace(candidate, takeoff_x=state.x, landing=result.landing)
        hop = self._snow_hop_path(state, candidates, index)
        if hop is None:
            self.log("snow_wait", x=round(state.x, 1), y=round(state.y, 1))
            self._snow_idle()
            return None
        return hop

    def _snow_idle(self) -> None:
        self.edge = None
        self.chain = None
        self.phase = "idle"
        self.cooldown_until = time.monotonic() + SNOW_WAIT_SECONDS

    def _wind_env_at(self, lead: float) -> StepEnv | None:
        start = self._wind_start(lead)
        if start is None:
            return None
        return StepEnv(wind=wind_velocity, start_time=start)

    def _verify_at(self, edge: Edge, x: float, y: float, lead: float) -> bool:
        return self._verified_at(edge, x, y, lead) is not None

    def _verified_at(self, edge: Edge, x: float, y: float, lead: float) -> JumpResult | None:
        """Edge verification with the wind phase projected `lead` seconds ahead
        (the moment the arc would really leave the ground)."""
        env = self._wind_env_at(lead) if self.wind_enabled else None
        if env is None:
            return self.edge_factory.verified(edge, x, y)
        saved = (self.edge_factory.env, self.edge_factory.wind_enabled, self.edge_factory.epoch)
        self.edge_factory.set_wind(env, True, saved[2])
        try:
            return self.edge_factory.verified(edge, x, y)
        finally:
            self.edge_factory.set_wind(*saved)

    def _execution_x(self, state: PlayerSaveState, edge: Edge) -> float:
        """Where the arc really starts when the charge begins right now: the
        current position plus the charge drift (wind summed over the charge
        frames; walk velocity for the first frame when wind is off)."""
        if self.wind_offset is not None and self.wind_enabled:
            return state.x + self._charge_drift(state, edge, 0.05)
        return self._launch_x(state)

    @staticmethod
    def _liftoff_lead(edge: Edge) -> float:
        """Seconds from now to the arc leaving the ground: the charge plus the
        input latency."""
        return 0.05 + edge.frames / 60.0

    def _charge_drift(self, state: PlayerSaveState, edge: Edge, lead: float) -> float:
        """Horizontal drift accumulated while the game charges the jump.

        On the ground the behaviour tree zeroes `velocity.X` every frame
        (except on ice), so while charging only the wind moves the king: each
        frame adds one wind step before the position update. The body also
        keeps its pre-charge (walk/wind) velocity for the first frame. `lead`
        is the wind-model time of the first charge frame.
        """
        drift = state.vx if abs(state.vx) > 0.01 else 0.0
        if not self.wind_enabled:
            return drift
        start = self._wind_start(lead)
        if start is None:
            return drift
        for index in range(edge.frames):
            drift += wind_velocity(start + index / 60.0)
        return drift

    @staticmethod
    def _wind_shift(tau: float, flight: int, charge: int) -> float:
        """Extra horizontal displacement caused by the wind for a launch at
        model time `tau`: one wind step per charge frame (velocity.X is zeroed
        while charging), then the flight, where each frame's velocity already
        carries the wind accumulated so far (`sum w_i * (frames - i)`)."""
        shift = 0.0
        for k in range(charge):
            shift += wind_velocity(tau - (k + 1) / 60.0)
        for i in range(flight):
            shift += wind_velocity(tau + i / 60.0) * (flight - i)
        return shift

    def _no_wind_crossing(self, edge: Edge, x: float, y: float) -> tuple[int, float] | None:
        """Frames from liftoff until the feet reach the target height on the
        way down, plus the no-wind hitbox x there. None when a wall or slope
        deflects the arc (the linear wind model would not hold)."""
        result = simulate_jump(self.grid, x, y, edge.direction, edge.frames)
        if not result.frames:
            return None
        target = self.platforms[edge.target]
        previous_vx = None
        for index, frame in enumerate(result.frames):
            if previous_vx is not None and abs(frame.vx - previous_vx) > 0.25:
                return None
            previous_vx = frame.vx
            if frame.y + PLAYER_HEIGHT >= target.top_y and frame.vy > 0:
                return index + 1, frame.x
        return None

    def _wind_launch_delay(self, edge: Edge, state: PlayerSaveState, y: float) -> float | None:
        """Seconds to idle at the takeoff until the wind makes `edge` land, or
        None when no launch time in the next wind period works.

        The wind function is known exactly, so the landing shift for a launch
        time is a closed-form integral of it; scanning the period costs
        milliseconds instead of hundreds of simulations. The candidate is then
        confirmed once with the exact simulator."""
        crossing = self._no_wind_crossing(edge, state.x, y)
        if crossing is None:
            return self._wind_launch_delay_sim(edge, state, y)
        flight, lx0 = crossing
        target = self.platforms[edge.target]
        lo = target.x0 - 16.0 + MIN_MARGIN - lx0
        hi = target.x1 - MIN_MARGIN - lx0
        period_s = 2.0 * math.pi / WIND_OMEGA
        start = self._wind_start(0.0)
        if start is None:
            return None
        charge_s = 0.05 + edge.frames / 60.0
        delay = 0.05
        step = 0.08
        while delay <= period_s:
            tau = start + delay + charge_s
            shift = self._wind_shift(tau, flight, edge.frames)
            if lo <= shift <= hi:
                launch_x = state.x + self._charge_drift(state, edge, delay)
                if self._verified_at(edge, launch_x, y, delay + charge_s):
                    return delay
            delay += step
        return None

    def _safe_landing_from(self, state: PlayerSaveState, edge: Edge) -> tuple[Edge, object] | None:
        """Any safe landing at all from where the king stands: the arc only has
        to land well inside a platform near the current height (there is no
        exact-spot requirement, the window planner re-plans after every
        landing)."""
        env = self._wind_env_at(self._liftoff_lead(edge)) if self.wind_enabled else None
        result = simulate_jump(
            self.grid,
            self._execution_x(state, edge),
            state.y,
            edge.direction,
            edge.frames,
            env=env,
            wind_enabled=env is not None,
        )
        if result.outcome != "landed" or result.landing is None:
            return None
        if env is not None and self.edge_factory._collision_kick(result):
            return None
        if result.landing[1] > state.y + PLAYER_HEIGHT + 60.0:
            return None
        platform = platform_below(self.platforms, result.landing[0], result.landing[1])
        if platform is None:
            return None
        margin = min(result.landing[0] - platform.x0, platform.x1 - result.landing[0])
        if margin < 5.0:
            return None
        index = self.chain_planner.index_of[id(platform)]
        return replace(edge, target=index, takeoff_x=state.x, landing=result.landing), platform

    def _wind_launch_delay_sim(self, edge: Edge, state: PlayerSaveState, y: float) -> float | None:
        """Fallback window search with full simulations (arcs that collide)."""
        charge_s = edge.frames / 60.0 + 0.05
        period_s = 2.0 * math.pi / WIND_OMEGA
        step = 0.15
        delay = 0.05
        while delay <= period_s:
            launch_x = state.x + self._charge_drift(state, edge, delay)
            if self._verify_at(edge, launch_x, y, delay + charge_s) and self._verify_at(
                edge, launch_x, y, delay + charge_s + 0.25
            ):
                return delay
            delay += step
        return None

    def _snow_hop_path(
        self,
        state: PlayerSaveState,
        candidates: list[Edge],
        index: int | None,
    ) -> Edge | None:
        """Breadth-first search over minimum-charge hops (both directions).

        Returns the first hop of the shortest sequence that lands somewhere an
        arc to the chain's target verifies from at the phase that time would
        have, or None. Each hop is simulated wind-aware with the factory env;
        the path is only a guide, because the controller re-plans after every
        hop.
        """
        start = (state.x, state.y)
        frontier: list[tuple[float, float]] = [start]
        seen: set[tuple[float, float]] = {start}
        parents: dict[tuple[float, float], tuple[tuple[float, float], int, JumpResult]] = {}
        for depth in range(SNOW_HOP_LOOKAHEAD):
            env = self._wind_env_at(SNOW_HOP_LEAD + depth * SNOW_HOP_SECONDS)
            nxt: list[tuple[float, float]] = []
            for node in frontier:
                for direction in (-1, 1):
                    result = simulate_jump(
                        self.grid,
                        node[0],
                        node[1],
                        direction,
                        SNOW_HOP_FRAMES,
                        env=env,
                        wind_enabled=env is not None,
                    )
                    if result.outcome != "landed" or result.landing is None:
                        continue
                    platform = platform_below(self.platforms, result.landing[0], result.landing[1])
                    if platform is not None and platform.width >= 24.0:
                        # A residual wind-phase error shifts a hop a few px;
                        # landings near an edge would be pushed off the
                        # platform before the next decision.
                        margin = min(
                            result.landing[0] - platform.x0,
                            platform.x1 - result.landing[0],
                        )
                        if margin < 8.0:
                            continue
                    landing = (result.landing[0], result.landing[1] - PLAYER_HEIGHT)
                    if landing in seen:
                        continue
                    seen.add(landing)
                    parents[landing] = (node, direction, result)
                    arc_lead = SNOW_HOP_LEAD + (depth + 1) * SNOW_HOP_SECONDS + SNOW_ARC_LEAD
                    if any(
                        self._verify_at(candidate, landing[0], landing[1], arc_lead)
                        for candidate in candidates
                    ):
                        return self._first_hop(start, landing, parents, index)
                    nxt.append(landing)
            frontier = nxt
            if not frontier:
                break
        return None

    def _first_hop(
        self,
        start: tuple[float, float],
        landing: tuple[float, float],
        parents: dict[tuple[float, float], tuple[tuple[float, float], int, JumpResult]],
        index: int | None,
    ) -> Edge:
        node = landing
        depth = 0
        while parents[node][0] != start:
            node = parents[node][0]
            depth += 1
        _parent, direction, result = parents[node]
        landed = platform_below(self.platforms, result.landing[0], result.landing[1])
        target = self.chain_planner.index_of[id(landed)] if landed is not None else index
        hop = Edge(
            source=index if index is not None else 0,
            target=target if target is not None else 0,
            takeoff_x=start[0],
            direction=direction,
            frames=SNOW_HOP_FRAMES,
            landing=result.landing,
            robust=False,
            clearance=0.0,
            margin=100.0,
            gain=0.0,
        )
        self.log(
            "snow_hop",
            x=round(start[0], 1),
            direction=direction,
            landing=(round(hop.landing[0], 1), round(hop.landing[1], 1)),
            target=round(self.platforms[hop.target].top_y, 1),
            path=depth + 1,
        )
        return hop

    def _walk_step(self, state: PlayerSaveState) -> None:
        self._ensure_focus()
        if self._on_snow(state.x, state.y) and abs(self.walk_target - state.x) > 1.0:
            self._blacklist_edge(state, reason="snow_walk")
            return
        if self._on_ice(state):
            self._walk_ice(state)
            return
        error = self.walk_target - state.x
        left = self.cfg.input.left_key
        right = self.cfg.input.right_key
        if abs(error) <= 1.0:
            self.keyboard.up(left)
            self.keyboard.up(right)
            self.phase = "takeoff"
            return
        now = time.monotonic()
        if self.walk_last_x is None:
            self.walk_last_x = state.x
            self.walk_last_move = now
        elif abs(state.x - self.walk_last_x) < 0.4:
            if now - self.walk_last_move > 1.2:
                self.keyboard.up(left)
                self.keyboard.up(right)
                self._blacklist_edge(state, reason="walk_blocked")
                return
        else:
            self.walk_last_x = state.x
            self.walk_last_move = now
        self.keyboard.up(left)
        self.keyboard.up(right)
        key = right if error > 0 else left
        direction = 1 if error > 0 else -1
        if abs(error) <= 5.0:
            max_frames = 1
        else:
            max_frames = min(4, max(1, int(round(abs(error) / WALK_SPEED))))
        frames = 0
        for candidate in range(max_frames, 0, -1):
            # The keyboard boundary can hold the key one frame longer than
            # requested, wind adds up to WIND_STRENGTH per frame, and walking
            # can overshoot the platform edge; only tap when the whole worst
            # case step keeps ground under the hitbox.
            reach = state.x + direction * (candidate + 1) * (WALK_SPEED + WIND_STRENGTH)
            if self._path_supported(state.x, reach, state.y):
                frames = candidate
                break
        if frames == 0:
            if abs(error) <= 1.0:
                self.phase = "takeoff"
                return
            edge = self.edge
            if edge is not None:
                # Walking closer is unsafe; jump from here when the arc still
                # verifies from the drifting position, or when it lands safely
                # somewhere anyway (the planner re-plans after landing).
                lead = 0.05 + edge.frames / 60.0
                if self._verified_at(edge, self._execution_x(state, edge), state.y, lead):
                    self.phase = "takeoff"
                    return
                safe = self._safe_landing_from(state, edge)
                if safe is not None:
                    self.edge = safe[0]
                    self.phase = "takeoff"
                    return
            self._blacklist_edge(state, reason="walk_edge")
            return
        self.frame_keyboard.tap_frames(key, frames)

    def _on_ice(self, state: PlayerSaveState) -> bool:
        from .gamephysics import ICE

        return self.grid.type_at(state.x, state.y + PLAYER_HEIGHT + 1.0) == ICE

    def _supported(self, x: float, y: float) -> bool:
        from .slopes import collision_info

        return collision_info(self.grid, x, y + 1.0).hit

    def _path_supported(self, x0: float, x1: float, y: float) -> bool:
        """Is the hitbox supported at every ~2px along the walk segment."""
        if not self._supported(x1, y):
            return False
        span = x1 - x0
        steps = max(1, int(abs(span) / 2.0))
        for index in range(1, steps):
            if not self._supported(x0 + span * index / steps, y):
                return False
        return True

    @staticmethod
    def _launch_x(state: PlayerSaveState) -> float:
        """The x the king really leaves the ground from: the body keeps its
        walk velocity for the first charge frame, before JumpState zeroes it,
        so the verify must include that one-frame drift."""
        return state.x + (state.vx if abs(state.vx) > 0.01 else 0.0)

    def _walk_ice(self, state: PlayerSaveState) -> None:
        """Ice walking: while a key is held the game accelerates vx toward
        1.5 at `ICE_FRICTION * 2` per frame; after release the slide decays at
        `ICE_FRICTION`. The coast distance is the exact discrete sum
        `v^2 / (2 * ICE_FRICTION) + v / 2` px, so hold only while the projected
        stop point is short of the target, then coast."""
        left = self.cfg.input.left_key
        right = self.cfg.input.right_key
        error = self.walk_target - state.x
        velocity = state.vx
        coast = velocity * velocity / (2.0 * ICE_FRICTION) + abs(velocity) / 2.0
        projected = state.x + math.copysign(coast, velocity) if velocity != 0.0 else state.x
        if abs(error) <= 1.0 and abs(velocity) <= 0.05:
            self.keyboard.up(left)
            self.keyboard.up(right)
            self.phase = "takeoff"
            return
        if abs(projected - self.walk_target) <= 1.0 and velocity != 0.0:
            # The coast brings the king to the target: let friction stop him.
            self.keyboard.up(left)
            self.keyboard.up(right)
            return
        key = right if error > 0 else left
        other = left if key == right else right
        self.keyboard.up(other)
        self.keyboard.down(key)

    def _takeoff(self, state: PlayerSaveState) -> None:
        edge = self.edge
        if edge is None:
            self.phase = "idle"
            return
        now = time.monotonic()
        wind_active = self.wind_offset is not None and self.wind_enabled
        if wind_active and now < self.takeoff_wait_until:
            # Waiting for the exact wind strength/direction the plan needs:
            # hold position against the drift. Never walk or blacklist the
            # route just because the takeoff slid while waiting. The phase is
            # "takeoff" so the loop comes straight back here, not through
            # planning, until the window opens.
            self.phase = "takeoff"
            self.plan_hold_x = edge.takeoff_x
            self._plan_hold(state)
            return
        if abs(edge.takeoff_x - state.x) > 1.25:
            # Not at the planned takeoff. In a wind screen the king is always
            # sliding, so standing still on one spot is impossible: jump from
            # here whenever the arc verifies from this drifting position.
            result = self._verified_at(
                edge, self._execution_x(state, edge), state.y, self._liftoff_lead(edge)
            )
            if result is None or result.landing is None:
                self._start_walk(edge.takeoff_x)
                return
            edge = replace(edge, takeoff_x=state.x, landing=result.landing)
            self.edge = edge
            self.takeoff_wait_until = 0.0
        check_x = self._execution_x(state, edge)
        verified = self._verified_at(edge, check_x, state.y, self._liftoff_lead(edge))
        if verified is None:
            if wind_active:
                delay = self._wind_launch_delay(edge, state, state.y)
                if delay is not None and delay > 0.05:
                    # The edge is good, just not at this instant: wait for the
                    # phase window the simulation proved instead of re-planning.
                    self.takeoff_wait_until = now + delay
                    self.log(
                        "wind_window",
                        wait=round(delay, 2),
                        x=round(state.x, 1),
                        frames=edge.frames,
                    )
                    return
                # No usable window: fall back to a jump that is proven safe
                # (lands well inside some platform) instead of re-planning.
                safe = self._safe_landing_from(state, edge)
                if safe is not None:
                    edge, platform = safe
                    self.edge = edge
                    self.log(
                        "safe_fallback",
                        x=round(state.x, 1),
                        target=round(platform.top_y, 1),
                        landing=(round(edge.landing[0], 1), round(edge.landing[1], 1)),
                    )
                    check_x = self._execution_x(state, edge)
                else:
                    # Not the edge's fault: the wind phase shifted under it.
                    # Re-plan with the fresh phase instead of blacklisting.
                    self.log("verify_wind", x=round(state.x, 1))
                    self.edge = None
                    self.chain = None
                    self.phase = "idle"
                    self.cooldown_until = now + 0.05
                    return
            else:
                safe = self._safe_landing_from(state, edge)
                if safe is None:
                    self._blacklist_edge(state, reason="verify_failed")
                    return
                edge, platform = safe
                self.edge = edge
                check_x = self._execution_x(state, edge)
                self.log(
                    "safe_fallback",
                    x=round(state.x, 1),
                    target=round(platform.top_y, 1),
                    landing=(round(edge.landing[0], 1), round(edge.landing[1], 1)),
                )
        if verified is not None and verified.landing is not None:
            # The verified landing is what will really happen from here; keep
            # the planned-landing bookkeeping in sync so it is not a "miss".
            if abs(verified.landing[0] - edge.landing[0]) > 0.01:
                edge = replace(edge, landing=verified.landing)
                self.edge = edge
        if wind_active and not self._path_supported(state.x, check_x, state.y):
            # The charge itself drifts the king (velocity.X is zeroed while
            # charging, so only the wind moves him): charging from a takeoff
            # this close to the edge slides him off before the jump.
            self.log("charge_edge", x=round(state.x, 1), drift=round(check_x - state.x, 2))
            self._blacklist_edge(state, reason="charge_edge")
            return
        if self._on_ice(state) and abs(state.vx) > 0.05:
            return
        self._ensure_focus()
        direction_key = None
        if edge.direction == 1:
            direction_key = self.cfg.input.right_key
        elif edge.direction == -1:
            direction_key = self.cfg.input.left_key
        self.expected_vy = self._expected_vy(state, edge)
        self.wind_gate = self.wind_enabled
        # The game samples the keyboard one frame after our key-event boundary,
        # so holding N boundaries charges N+1 frames; request one less.
        request_frames = max(1, edge.frames - 1)
        ok = self.frame_keyboard.charge_jump(self.cfg.input.jump_key, direction_key, request_frames)
        self.stats.jumps += 1
        self.planned_landing = edge.landing
        self.planned_arc = [
            (frame.x + 9.0, frame.y + PLAYER_HEIGHT) for frame in self._arc(edge, state)
        ]
        self.takeoff_pos = (state.x, state.y)
        stamp = self.clock.stamp()
        self.takeoff_stamp = stamp if stamp is not None else state.time_stamp
        self.bad_liftoff = False
        self.flight_trace = []
        self.log(
            "jump",
            direction=edge.direction,
            frames=edge.frames,
            takeoff=(round(state.x, 1), round(state.y, 1)),
            planned_takeoff=round(edge.takeoff_x, 1),
            target=edge.landing,
            ok=ok,
        )
        self.phase = "flight"
        self.cooldown_until = time.monotonic() + 0.5

    def _expected_vy(self, state: PlayerSaveState, edge: Edge) -> float:
        intensity = charge_intensity(edge.frames)
        if self._on_snow(state.x, state.y) and intensity < 0.3:
            intensity = 0.3
        return -JUMP_SPEED * intensity

    def _arc(self, edge: Edge, state: PlayerSaveState) -> list:
        result = simulate_jump(
            self.grid,
            state.x,
            state.y,
            edge.direction,
            edge.frames,
            env=self.edge_factory.env,
            wind_enabled=self.edge_factory.wind_enabled,
        )
        return result.frames

    def _blacklist_edge(self, state: PlayerSaveState, reason: str = "") -> None:
        edge = self.edge
        if edge is not None:
            self.blacklist.add((round(edge.takeoff_x, 1), edge.direction, edge.frames))
        self.log("blacklist", reason=reason, x=round(state.x, 1))
        self.edge = None
        self.chain = None
        self.phase = "idle"
        self.takeoff_wait_until = 0.0
        self.cooldown_until = time.monotonic() + 0.3

    # -------------------------------------------------------------- flight

    def _check_liftoff(self, state: PlayerSaveState) -> None:
        """The first airborne samples can be many frames after liftoff (the
        control loop is slower than the game), so the allowed window is wide:
        velocity decays toward zero by gravity * frames."""
        if self.bad_liftoff or self.expected_vy is None:
            return
        if state.vy >= 0.0:
            return
        frames = max(0, state.time_stamp - self.takeoff_stamp) + 3
        expected = self.expected_vy + 0.257142857 * frames
        if state.vy > expected + 0.5 or state.vy < self.expected_vy - 0.35:
            self.bad_liftoff = True
            self.log(
                "bad_liftoff",
                vy=round(state.vy, 3),
                expected=round(self.expected_vy, 3),
                frames=frames,
            )

    def _verify_landing(self, state: PlayerSaveState) -> None:
        self._harvest_wind_deltas(state)
        planned = self.planned_landing
        edge = self.edge
        feet = (state.x, state.y + PLAYER_HEIGHT)
        if planned is None:
            self._reset_after_flight()
            return
        if self.takeoff_pos is not None:
            # The direction key can drift the king ~1.5px during the charge,
            # so a strict takeoff box would misread the still-grounded pre-jump
            # frame as a completed miss and blacklist a good edge.
            at_takeoff = (
                abs(state.x - self.takeoff_pos[0]) < 4.0
                and abs(state.y - self.takeoff_pos[1]) < 2.0
            )
            if at_takeoff and state.time_stamp - self.takeoff_stamp < 40:
                return
            if at_takeoff and state.time_stamp - self.takeoff_stamp >= 40:
                self.stats.misses += 1
                self.log("no_liftoff", x=round(state.x, 1))
                self._reset_after_flight()
                return
        dx = abs(feet[0] - planned[0])
        dy = abs(feet[1] - planned[1])
        self.last_landing_error = (dx, dy)
        off = dx > LANDING_TOLERANCE or dy > LANDING_TOLERANCE
        platform = platform_below(self.platforms, feet[0], feet[1])
        safe = False
        if platform is not None:
            margin = min(feet[0] - platform.x0, platform.x1 - feet[0])
            safe = margin >= 5.0
        if off and not safe:
            self.log(
                "landing",
                planned=planned,
                actual=feet,
                error=(round(dx, 1), round(dy, 1)),
                liftoff_ok=not self.bad_liftoff,
                trace=self.flight_trace or None,
            )
            self.stats.misses += 1
            if edge is not None:
                self.blacklist.add((round(edge.takeoff_x, 1), edge.direction, edge.frames))
            if feet[1] > planned[1] + 100.0:
                self.stats.falls += 1
                self.log("fall", y=round(state.y, 1))
        else:
            # Landing anywhere safely on a platform is fine: the rolling
            # window re-plans from wherever the king ended up.
            self.log(
                "landing",
                planned=planned,
                actual=feet,
                error=(round(dx, 1), round(dy, 1)),
                safe=safe and off,
            )
        self._reset_after_flight()

    def _reset_after_flight(self) -> None:
        self.edge = None
        self.chain = None
        self.planned_landing = None
        self.takeoff_pos = None
        self.expected_vy = None
        self.phase = "idle"
        self.takeoff_wait_until = 0.0
        self.cooldown_until = time.monotonic() + 0.12

    # ------------------------------------------------------------- recovery

    def _recover_from_splat(self, state: PlayerSaveState) -> None:
        """The splat fail state waits for a key press before the king gets up.

        See FailState.MyRun in the decompile: once the timer is done, any of
        left/right/jump releases the king. Tap toward the platform centre so
        the same key press cannot walk him off a ledge.
        """
        if self.phase == "flight" and self.flight_trace:
            self._harvest_wind_deltas(state)
            self.log(
                "flight_lost", x=round(state.x, 1), y=round(state.y, 1), trace=self.flight_trace
            )
        self.edge = None
        self.chain = None
        self.planned_landing = None
        if self.phase in ("walk", "walk_off"):
            self.keyboard.release_all()
        self.phase = "splat"
        if not state.grounded or time.monotonic() < self.splat_tap_at:
            return
        self.splat_tap_at = time.monotonic() + SPLAT_TAP_SECONDS
        platform = platform_below(self.platforms, state.x, state.y + PLAYER_HEIGHT)
        center = (platform.x0 + platform.x1) / 2.0 if platform is not None else state.x + 9.0
        key = self.cfg.input.right_key if state.x + 9.0 < center else self.cfg.input.left_key
        self.frame_keyboard.tap_frames(key, 2)
        self.log("splat_recover", x=round(state.x, 1), y=round(state.y, 1), key=key)

    def _recover(self, state: PlayerSaveState) -> None:
        """No chain: search for any verified upward option on the worker, so
        the control loop can keep holding the king while it computes."""
        now = time.monotonic()
        if now - self.stats.progress_at < 3.0:
            self.cooldown_until = now + 0.5
            return
        self.stats.progress_at = now
        self.plan_hold_x = state.x
        self._recover_result = None
        self._recover_thread = threading.Thread(
            target=self._recover_worker, args=(state.x, state.y), daemon=True
        )
        self._recover_thread.start()

    def _recover_worker(self, x: float, y: float) -> None:
        try:
            self._recover_result = options_for(self.grid, self.platforms, x, y, include_lower=False)
        except Exception as exc:
            self._recover_result = None
            self.log("recover_error", error=repr(exc))

    def _finish_recover(self, state: PlayerSaveState, options: list[Option]) -> None:
        options = [o for o in options if o.key not in self.blacklist]
        self.log("stuck", x=round(state.x, 1), y=round(state.y, 1), options=len(options))
        if options:
            self.last_options = options
            chosen = options[0]
            self.last_chosen = chosen
            edge = Edge(
                source=0,
                target=0,
                takeoff_x=chosen.takeoff_x,
                direction=chosen.direction,
                frames=chosen.frames,
                landing=chosen.landing,
                robust=chosen.robust,
                clearance=chosen.clearance,
                margin=chosen.margin,
                gain=chosen.gain,
            )
            self.edge = edge
            self.chain = None
            if abs(chosen.takeoff_x - state.x) > 0.75:
                self._start_walk(chosen.takeoff_x)
            else:
                self._takeoff(state)
            return
        run_x0, run_x1 = surface_run(self.grid, state.x, state.y)
        falls: list[tuple[float, float, float]] = []
        for edge_x, walk_direction in ((run_x0 - 18.0, -1.0), (run_x1, 1.0)):
            if edge_x == run_x1 and run_x1 > 471.0:
                continue
            landing = simulate_fall(self.grid, edge_x, state.y, vx=walk_direction * WALK_SPEED)
            if landing is None:
                continue
            if landing[1] <= state.y + 26.0:
                continue
            falls.append((edge_x, landing[0], landing[1]))
        if falls:
            edge_x, lx, ly = max(falls, key=lambda item: item[2])
            self.planned_landing = (lx, ly)
            self.planned_arc = []
            self.takeoff_pos = None
            self.edge = None
            self.log("walk_off", edge=round(edge_x, 1), landing=(round(lx, 1), round(ly, 1)))
            target = edge_x - 7.0 if edge_x < state.x else edge_x + 7.0
            self._start_walk(target)
            self.phase = "walk_off"
            return
        run_x0, run_x1 = surface_run(self.grid, state.x, state.y)
        mid = (run_x0 + run_x1) / 2.0
        if abs(mid - state.x) > 12.0:
            self._start_walk(mid)
        else:
            self.cooldown_until = time.monotonic() + 1.0

    # ---------------------------------------------------------------- loop

    def step(self) -> None:
        state = self._read()
        if state is None:
            time.sleep(0.02)
            return
        self.state = state
        if time.monotonic() >= self.deadline:
            return
        screen = state.screen_index
        if screen != self.screen:
            self.screen = screen
            self.log("screen", screen=screen, x=round(state.x, 1), y=round(state.y, 1))
            if self.logger is not None and self.current_frame is not None:
                self.logger.frame(self.current_frame, f"screen_{screen:03d}")
        if state.grounded and state.y < self.stats.best_y - 4.0:
            self.stats.best_y = state.y
            self.stats.best_screen = screen
            self.stats.progress_at = time.monotonic()
            self.log("best", y=round(state.y, 1), x=round(state.x, 1), screen=screen)
            self._dump_ui_label = "best"
        self._sample_standing_wind(state)
        if self.ending_detected:
            return
        if screen >= GOAL_SCREEN and state.grounded and state.x + 9.0 >= GOAL_CENTER_X:
            self.ending_detected = True
            self.log("ending_reached", screen=screen, x=round(state.x, 1))
            return
        if state.state == SPLAT_STATE:
            self._recover_from_splat(state)
            return
        if not state.grounded:
            if self.phase in ("walk", "walk_off"):
                self.keyboard.release_all()
                self.walk_last_x = None
                if self.phase == "walk":
                    self.log(
                        "walk_fell",
                        x=round(state.x, 1),
                        target=round(self.walk_target, 1),
                    )
                    self.stats.falls += 1
                    self.stats.misses += 1
                    self.log("fall", y=round(state.y, 1), reason="walk")
                    self._blacklist_edge(state, reason="fell_walking")
                else:
                    self.phase = "flight"
            if self.phase == "flight":
                if len(self.flight_trace) < 400:
                    self.flight_trace.append(
                        (
                            state.time_stamp,
                            round(state.x, 2),
                            round(state.y, 2),
                            round(state.vx, 3),
                            round(state.vy, 3),
                        )
                    )
                self._check_liftoff(state)
            return
        if self.phase == "flight":
            self._verify_landing(state)
            return
        if self.phase in ("walk", "walk_off"):
            self._walk_step(state)
            return
        if self.phase == "takeoff":
            self._takeoff(state)
            return
        if time.monotonic() < self.cooldown_until:
            return
        if self.blacklist and time.monotonic() > self.blacklist_until:
            self.blacklist.clear()
            self.blacklist_until = time.monotonic() + 25.0
        if self.plan_idle():
            # Never swap the factory's wind env under a running plan worker.
            self._configure_wind(state)
        self._decide(state)

    def run(self, max_minutes: float | None = None, ui=None) -> RunStats:
        limit = max_minutes if max_minutes is not None else self.cfg.play.max_minutes
        deadline = time.monotonic() + limit * 60.0
        self.deadline = deadline
        self.edge_factory.deadline = deadline
        kill_key = self.cfg.play.kill_key
        self.stats.started_at = time.monotonic()
        self.stats.progress_at = time.monotonic()
        self.blacklist_until = time.monotonic() + 25.0
        from .capture import GameView, is_menu_frame

        focus(self.window.hwnd)
        hint = vision_hint(self.window)
        self.memory.attach()
        self.memory.ensure_address(*(hint or (None, None)), timeout_s=20.0)
        if self.memory.address is not None:
            self.log("memory_ready", address=hex(self.memory.address))
        # The run-timer cache belongs to the game process: a restarted game
        # leaves a stale address that would silently disable the wind model,
        # so validate it and refit the phase against the new timer baseline.
        if self.memory.pm is not None and self.run_timer.address is not None:
            if not self.run_timer.verify(self.memory.pm):
                self.log("run_timer_stale", address=hex(self.run_timer.address))
                self.run_timer.address = None
                self.wind_offset = None
                self.wind_validated = False
                # Samples belong to the old timer baseline; mixing them would
                # produce an offset that is wrong by whole phase steps.
                self.wind_deltas.clear()
                self.wind_fit_count = 0
        if self.run_timer.address is None and self.run_timer.find(self.memory.pm, timeout_s=8.0):
            self.log("run_timer", address=hex(self.run_timer.address))
        interval = 1.0 / 60.0
        try:
            sys.setswitchinterval(0.002)
            with GameView(self.window) as view:
                view.calibrate_viewport()
                perf_at = time.monotonic()
                perf_loops = 0
                perf_step = perf_grab = perf_ui = 0.0
                while time.monotonic() < deadline:
                    loop_start = time.monotonic()
                    if key_is_down(kill_key):
                        self.log("abort", reason="kill key")
                        break
                    if ui is not None and getattr(ui, "abort_requested", False):
                        self.log("abort", reason="stop button")
                        break
                    if ui is not None and getattr(ui, "paused", False):
                        self.keyboard.release_all()
                        self.current_frame = view.grab(ensure_focus=False)
                        self._update_ui(ui)
                        time.sleep(0.05)
                        continue
                    self._frame_counter += 1
                    grabbed_at = time.monotonic()
                    # The controller plans from memory; pixels only feed the
                    # overlay and the menu check, so 15 Hz is plenty and saves
                    # ~20ms per frame in capture+resize.
                    if self._frame_counter % 4 == 0:
                        self.current_frame = view.grab(ensure_focus=False)
                    stepped_at = time.monotonic()
                    # A black background means the game is back at the main
                    # menu: stop the bot instead of playing the menu. Two
                    # fresh frames in a row avoid fade-frame false positives.
                    if (
                        self._frame_counter % 4 == 0
                        and self.current_frame is not None
                        and is_menu_frame(self.current_frame)
                    ):
                        self._menu_hits += 1
                        if self._menu_hits >= 2:
                            self.log("menu_stop", frame=self._frame_counter)
                            break
                    elif self._frame_counter % 4 == 0:
                        self._menu_hits = 0
                    self.step()
                    draw_at = time.monotonic()
                    if self._dump_ui_label is not None and ui is not None and self.logger:
                        ui.save_async(str(self.logger.dir / f"{self._dump_ui_label}.png"))
                        self._dump_ui_label = None
                    self._update_ui(ui)
                    drawn_at = time.monotonic()
                    perf_loops += 1
                    perf_grab += stepped_at - grabbed_at
                    perf_step += draw_at - stepped_at
                    perf_ui += drawn_at - draw_at
                    if drawn_at - perf_at >= 5.0:
                        span = drawn_at - perf_at
                        self.log(
                            "perf",
                            hz=round(perf_loops / span, 1),
                            step_ms=round(1000.0 * perf_step / perf_loops, 2),
                            grab_ms=round(1000.0 * perf_grab / perf_loops, 2),
                            ui_ms=round(1000.0 * perf_ui / perf_loops, 2),
                            planning=not self.plan_idle(),
                        )
                        perf_at, perf_loops = drawn_at, 0
                        perf_step = perf_grab = perf_ui = 0.0
                    elapsed = drawn_at - loop_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
        finally:
            self.keyboard.release_all()
            for thread in (self._plan_thread, self._recover_thread):
                if thread is not None:
                    thread.join(timeout=1.0)
            self.edge_factory.save_cache()
            if ui is not None and self.logger is not None:
                ui.save(str(self.logger.dir / "ui_final.png"))
            if self.logger is not None:
                self.logger.event("stats", **self.stats.summary())
        return self.stats

    def _update_ui(self, ui) -> None:
        if ui is None or self.current_frame is None:
            return
        ui.update(self)
        ui.show()
