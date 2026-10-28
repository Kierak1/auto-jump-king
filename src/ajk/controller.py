"""Main control loop: perception -> planning -> decision -> execution.

The heavy lifting lives in the mixin modules:
  tracking.py    memory/vision position tracking and camera flips
  perception.py  collision sensing and screen identification
  planning.py    takeoff search and jump candidate generation
  execution.py   key presses, walking and landing bookkeeping
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

from .actions import Keyboard, key_is_down
from .brain import (
    Brain,
    BrainDecision,
    Candidate,
    make_brain,
)
from .capture import GameView
from .collision import CollisionMap
from .config import Config
from .decision import DecisionMixin
from .execution import ExecutionMixin, precise_sleep
from .level import LevelAtlas, ScreenIdentifier
from .memory import PlayerMemory
from .perception import PerceptionMixin
from .physics import JumpPlan, PhysicsParams
from .planning import PlanningMixin
from .state import ControllerStats, PlayerState, TrackPoint
from .telemetry import RunLogger
from .tracking import TrackingMixin
from .vision import (
    CameraTracker,
    GroundSensor,
    PlayerDetector,
    PlayerObs,
    SurfaceScanner,
)
from .visual import VisualSolidClassifier
from .win import WindowInfo

__all__ = ["Controller", "precise_sleep"]


class Controller(
    TrackingMixin,
    PerceptionMixin,
    PlanningMixin,
    ExecutionMixin,
    DecisionMixin,
):
    def __init__(
        self,
        cfg: Config,
        window: WindowInfo,
        brain: Brain | None = None,
        logger: RunLogger | None = None,
    ) -> None:
        self.cfg = cfg
        self.window = window
        self.params = PhysicsParams(
            gravity=cfg.physics.gravity,
            air_max_speed=cfg.physics.air_max_speed,
            max_charge_s=cfg.physics.max_charge_s,
            terminal_speed=cfg.physics.terminal_speed,
            charge_v0=[(float(c), float(v)) for c, v in cfg.physics.charge_v0],
        )
        self.brain = brain or make_brain(cfg.brain)
        self.logger = logger
        self.detector = PlayerDetector()
        self.keyboard = Keyboard(tap_ms=cfg.input.tap_ms)
        game_dir = Path(cfg.game_dir)
        self.atlas = LevelAtlas(game_dir)
        self.screen_identifier = ScreenIdentifier(game_dir)
        self.collision = CollisionMap(atlas=self.atlas)
        self.last_screen_check = 0.0
        self.screen_cell: tuple[int, int] | None = None
        self.screen_changed_at = 0.0
        self.screen_mismatch_count = 0
        self.history: deque[TrackPoint] = deque(maxlen=90)
        self.state = PlayerState()
        self.stats = ControllerStats()
        self.prev_obs: PlayerObs | None = None
        self.last_ground_y: float | None = None
        self.jump_started_at = 0.0
        self.cooldown_until = 0.0
        self.last_decision_at = 0.0
        self.last_choice = ""
        self.current_plan: JumpPlan | None = None
        self.consecutive_misses = 0
        self.recent_falls: deque[float] = deque(maxlen=20)
        self.airborne = False
        self.was_falling = False
        self.stable_frames = 0
        self.pending: tuple[JumpPlan | None, Candidate | None] | None = None
        self.walk_target: float | None = None
        self.walk_started_at = 0.0
        self.walk_start_x = 0.0
        self.last_decision: BrainDecision | None = None
        self.last_candidates: list[tuple[Candidate, JumpPlan]] = []
        self.last_plan: JumpPlan | None = None
        self.last_state_text = ""
        self.decision_latency = 0.0
        self.classifier = VisualSolidClassifier()
        self.ground_sensor = GroundSensor()
        self.surface_scanner = SurfaceScanner(
            brightness=cfg.vision.surface_brightness,
            below_max=cfg.vision.surface_below_max,
            min_run=cfg.vision.surface_min_run,
        )
        self.classifier_solid: set[tuple[int, int]] = set()
        self.classifier_free: set[tuple[int, int]] = set()
        self.surface_cells: set[tuple[int, int]] = set()
        self.surface_hits: dict[tuple[int, int], int] = {}
        self.ui: object | None = None
        self.tracker: CameraTracker | None = None
        self.view_offset: tuple[float, float] = (0.0, 0.0)
        self.recent_failures: deque[tuple[float, float, float, float]] = deque(maxlen=40)
        self.last_key_refresh = 0.0
        self.verbose = False
        self.last_candidate: Candidate | None = None
        self.memory = PlayerMemory()
        self.memory_ok = False
        self.last_memory_pos: tuple[float, float] | None = None
        self.last_memory_attempt = 0.0
        self.player_visible = False
        self.replan = False
        self.intended_landing: tuple[float, float] | None = None
        self.memory_vision_drift = 0.0
        self.blocked_takeoffs: deque[tuple[float, float]] = deque(maxlen=40)
        self.jump_liftoff_y: float | None = None
        self.jump_takeoff_x = 0.0
        self.jump_lifted = False
        self.last_flight_log = 0.0
        self.last_escape_at = 0.0
        self.best_level_y = 1e9
        self.last_progress_at = 0.0
        self._dump_ui_label: str | None = None
        self.jump_apex_dumped = False
        self.last_landing_result: (
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
        ) = None

    def _now(self) -> float:
        return time.monotonic()

    def _ensure_focus(self) -> None:
        from .win import focus, is_foreground

        if not is_foreground(self.window.hwnd):
            focus(self.window.hwnd)

    def step(self, view: GameView, tracker: CameraTracker) -> bool:
        self.tracker = tracker
        frame = view.grab()
        obs = self.detector.detect(frame, self.prev_obs)
        self.player_visible = obs is not None
        if obs is not None:
            self.consecutive_misses = 0
            self.prev_obs = obs
            mask = [
                (
                    int(obs.left) - 2,
                    int(obs.top) - 2,
                    int(obs.right) + 2,
                    int(obs.bottom) + 2,
                )
            ]
            tracker.update(frame, mask)
        else:
            self.consecutive_misses += 1
            tracker.update(frame, [])
            if not self.memory_ok and self.consecutive_misses > 90:
                raise RuntimeError("lost track of the player")
        if not self.memory_ok:
            tracker.realign(0.0, 0.0)
        self.view_offset = (tracker.offset_x, tracker.offset_y)
        if obs is not None:
            self._learn_collision(obs, tracker.offset_x, tracker.offset_y)
        screen_changed = self._update_screen(frame)
        self._update_state(obs, tracker.offset_x, tracker.offset_y)
        if self.state.grounded:
            row = self.screen_cell[0] if self.screen_cell is not None else 0
            level_y = self.state.y - row * 360.0
            if level_y < self.best_level_y - 4.0:
                self.best_level_y = level_y
                self.last_progress_at = self._now()
                self._dump_ui_label = "best"
                if self.logger:
                    self.logger.event(
                        "best",
                        level_y=round(level_y, 1),
                        x=round(self.state.x, 1),
                        cell=list(self.screen_cell or ()),
                    )
        if screen_changed and self.logger:
            vision = (
                (obs.cx + tracker.offset_x, obs.bottom + tracker.offset_y)
                if obs is not None
                else None
            )
            self.logger.event(
                "screen_change",
                cell=list(self.screen_cell or ()),
                memory=self.memory.read(),
                vision=vision,
                tracker=[tracker.offset_x, tracker.offset_y],
            )
        self._sense_ground(frame, tracker.offset_x, tracker.offset_y)
        self._refresh_surfaces(frame, tracker.offset_x, tracker.offset_y)
        if self._now() - self.last_key_refresh > 0.25:
            self.keyboard.sync(
                {
                    self.cfg.input.left_key,
                    self.cfg.input.right_key,
                    self.cfg.input.jump_key,
                }
            )
            self.last_key_refresh = self._now()

        if self.current_plan is not None:
            if self.airborne:
                self.jump_lifted = True
            if self._now() - self.jump_started_at > 4.0:
                if self.verbose:
                    print("[jump] timeout, forcing landing bookkeeping")
                if self.logger:
                    self.logger.event("jump_timeout")
                self._finish_jump()
            elif self.jump_lifted:
                if self.state.grounded:
                    self._finish_jump()
                else:
                    if not self.jump_apex_dumped and self.state.vy >= 0:
                        self.jump_apex_dumped = True
                        self._dump_ui_label = f"jump_{self.stats.jumps:02d}_apex"
                    self._log_flight()
                    self._flight_control()
            elif self._now() - self.jump_started_at > 0.5:
                self._jump_failed()
            else:
                self._flight_control()
            self._maybe_update_ui(frame)
            return True

        if self.pending is not None:
            self._walk_step()
            self._maybe_update_ui(frame)
            return True

        if not self.state.grounded or (self._now() < self.cooldown_until and not self.replan):
            self._maybe_update_ui(frame)
            return True
        self.replan = False

        if self._now() - self.last_progress_at > 25.0:
            self.last_progress_at = self._now()
            self.blocked_takeoffs.clear()
            if self.verbose:
                print("[stuck] clearing blocked takeoffs")
            if self.logger:
                self.logger.event("stuck", x=round(self.state.x, 1), y=round(self.state.y, 1))

        if (
            not self.memory_ok
            and self.memory.available
            and obs is not None
            and not obs.merged
            and self._now() - self.last_memory_attempt > 30.0
        ):
            self.last_memory_attempt = self._now()
            self._setup_memory(view, tracker)
            if self.memory_ok:
                self.last_screen_check = 0.0
                self._maybe_update_ui(frame)
                return True

        self._refresh_visual_map(frame, tracker.offset_x, tracker.offset_y)
        candidates = self._candidates()
        self.last_candidates = candidates
        if not candidates:
            self.keyboard.release_all()
            if self.state.grounded and self._now() - self.last_escape_at > 1.0:
                self.last_escape_at = self._now()
                escape = self._escape_direction()
                if escape == 0.0:
                    escape = 24.0 if self.state.x < 240.0 else -24.0
                if self.verbose:
                    print(f"[escape] no candidates, walking {escape:+.0f}px")
                if self.logger:
                    self.logger.event("escape", x=round(self.state.x, 1), step=escape)
                self.pending = (None, None)
                self.walk_target = self.state.x + escape
                self.walk_started_at = self._now()
                self.walk_start_x = self.state.x
                self._walk_step()
            return True

        decision = self._ask_brain(candidates)
        self.stats.decisions += 1
        mapping = {c.key: (c, p) for c, p in candidates}
        chosen = mapping.get(decision.choice)
        if chosen is None or decision.confidence < self.cfg.brain.min_confidence:
            if chosen is None:
                self.stats.low_confidence += 1
            fallback = max(
                candidates,
                key=lambda cp: (
                    cp[0].kind == "jump",
                    cp[0].known_landing,
                    cp[0].dy,
                ),
            )
            chosen = fallback
            decision = BrainDecision(
                choice=fallback[0].key,
                confidence=decision.confidence,
                backend=f"{decision.backend}+fallback",
            )
        cand, plan = chosen
        if cand.dy <= 0 and abs(cand.dx) < 20:
            self.recent_failures.append(
                (
                    round(self.state.x + cand.walk_offset, 1),
                    round(cand.charge_s, 2),
                    float(cand.direction),
                    self._now(),
                )
            )
        if self.intended_landing is not None and plan.landing is not None:
            near = [
                (c, p)
                for c, p in candidates
                if p.landing is not None
                and abs(p.landing[0] - self.intended_landing[0]) < 24
                and abs(p.landing[1] - self.intended_landing[1]) < 24
                and abs(c.walk_offset) < 24
                and self._takeoff_clear(self.state.x + c.walk_offset, self.state.y)
                and not self._takeoff_blocked(self.state.x + c.walk_offset)
            ]
            if near:
                near.sort(key=lambda cp: abs(cp[0].walk_offset))
                cand, plan = near[0]
                decision = BrainDecision(
                    choice=cand.key,
                    confidence=decision.confidence,
                    backend=f"{decision.backend}+intent",
                )
        self.intended_landing = None
        if self.verbose:
            print(
                f"[decision] {decision.backend} choice={decision.choice} "
                f"conf={decision.confidence:.2f} candidates={len(candidates)} "
                f"latency={self.decision_latency * 1000:.0f}ms"
            )
        if self.logger:
            self.logger.event(
                "decision",
                backend=decision.backend,
                choice=decision.choice,
                confidence=decision.confidence,
                latency_s=round(self.decision_latency, 3),
                candidates=len(candidates),
                state={"x": self.state.x, "y": self.state.y},
                raw=decision.raw,
            )
        self.last_choice = decision.choice
        if abs(cand.walk_offset) >= 10:
            self.pending = (plan, cand)
            self.walk_target = self.state.x + cand.walk_offset
            self.walk_started_at = self._now()
            self.walk_start_x = self.state.x
            self.intended_landing = plan.landing
            if self.verbose:
                print(f"[walk] start -> x={self.walk_target:.0f} (offset {cand.walk_offset:+.0f})")
            self._walk_step()
        elif not self._takeoff_clear(self.state.x, self.state.y) or self._takeoff_blocked(
            self.state.x
        ):
            escape = self._escape_direction()
            if escape != 0:
                self.pending = (plan, cand)
                self.walk_target = self.state.x + escape
                self.walk_started_at = self._now()
                self.walk_start_x = self.state.x
                self.intended_landing = plan.landing
                if self.verbose:
                    print(f"[walk] blocked takeoff, escaping to x={self.walk_target:.0f}")
                self._walk_step()
            else:
                self.keyboard.release_all()
                self.cooldown_until = self._now() + 0.2
        else:
            self._execute_jump(plan, cand)
        self._maybe_update_ui(frame)
        return True

    def _maybe_update_ui(self, frame: np.ndarray) -> None:
        if self.ui is None:
            return
        self.ui.update(frame, self)
        self.ui.show()
        if self._dump_ui_label is not None and self.logger is not None:
            self.ui.save(str(self.logger.dir / f"{self._dump_ui_label}.png"))
            self._dump_ui_label = None

    def dump_ui(self, path: str) -> None:
        if self.ui is not None:
            self.ui.save(path)

    def run(
        self,
        max_minutes: float | None = None,
        ui: object | None = None,
    ) -> ControllerStats:
        limit = max_minutes if max_minutes is not None else self.cfg.play.max_minutes
        deadline = self._now() + limit * 60.0
        kill_key = self.cfg.play.kill_key
        interval = 1.0 / max(self.cfg.play.fps, 1)
        from .win import focus

        self.ui = ui
        self.last_progress_at = self._now()
        focus(self.window.hwnd)
        with GameView(self.window) as view:
            view.calibrate_viewport()
            first = view.grab()
            tracker = CameraTracker(first.shape[:2])
            self.tracker = tracker
            self._setup_memory(view, tracker)
            try:
                while self._now() < deadline:
                    loop_start = self._now()
                    if key_is_down(kill_key):
                        if self.logger:
                            self.logger.event("abort", reason="kill key")
                        break
                    if self.ui is not None and getattr(self.ui, "abort_requested", False):
                        if self.logger:
                            self.logger.event("abort", reason="stop button")
                        break
                    if self.ui is not None and getattr(self.ui, "paused", False):
                        self.keyboard.release_all()
                        self._maybe_update_ui(view.grab())
                        time.sleep(0.05)
                        continue
                    if not view.is_focused():
                        view.focus()
                    try:
                        self.step(view, tracker)
                    except RuntimeError as exc:
                        if self.logger:
                            self.logger.event("error", message=str(exc))
                        break
                    elapsed = self._now() - loop_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
            finally:
                self.keyboard.release_all()
                if self.ui is not None:
                    self.dump_ui(str(self.logger.dir / "ui_final.png")) if self.logger else None
                    self.ui.close()
                if self.logger:
                    self.logger.event("stats", **self.stats.summary())
        return self.stats
