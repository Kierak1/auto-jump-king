"""Jump and walk execution plus landing bookkeeping.

Expects these attributes on the controller: cfg, keyboard, state, logger,
current_plan, last_plan, last_candidate, jump_started_at, jump_liftoff_y,
jump_takeoff_x, jump_lifted, jump_apex_dumped, last_flight_log, stats,
cooldown_until, recent_failures, recent_falls, blocked_takeoffs, last_landing_result,
pending, walk_target, walk_started_at, walk_start_x, replan, verbose, _dump_ui_label.
"""

from __future__ import annotations

import time

from .brain import Candidate
from .physics import JumpPlan


def precise_sleep(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)


class ExecutionMixin:
    def _execute_jump(self, plan: JumpPlan, cand: Candidate) -> None:
        self._ensure_focus()
        direction_key = self.cfg.input.right_key if plan.direction > 0 else self.cfg.input.left_key
        if plan.direction != 0 and plan.drift == "hold":
            self.keyboard.down(direction_key)
            time.sleep(0.04)
        self.keyboard.down(self.cfg.input.jump_key)
        precise_sleep(plan.charge_s)
        self.keyboard.up(self.cfg.input.jump_key)
        self.jump_started_at = self._now()
        self.current_plan = plan
        self.last_plan = plan
        self.last_candidate = cand
        self.jump_liftoff_y = self.state.y
        self.jump_takeoff_x = self.state.x
        self.jump_lifted = False
        self.jump_apex_dumped = False
        self.last_flight_log = 0.0
        self.stats.jumps += 1
        if self.verbose:
            target = plan.landing or (0.0, 0.0)
            print(
                f"[jump] dir={plan.direction:+d} drift={plan.drift} "
                f"charge={plan.charge_s:.2f}s walk={cand.walk_offset:+.0f} "
                f"from=({self.state.x:.0f},{self.state.y:.0f}) "
                f"target=({target[0]:.0f},{target[1]:.0f}) "
                f"known={plan.landing_known}"
            )
        if self.logger:
            self.logger.event(
                "jump",
                direction=plan.direction,
                drift=plan.drift,
                charge_s=plan.charge_s,
                v0=plan.v0,
                target=plan.landing,
                known=plan.landing_known,
                walk_offset=cand.walk_offset,
                from_=(round(self.state.x, 1), round(self.state.y, 1)),
            )

    def _walk_step(self) -> None:
        if self.walk_target is None or self.pending is None:
            return
        self._ensure_focus()
        left_key = self.cfg.input.left_key
        right_key = self.cfg.input.right_key
        error = self.walk_target - self.state.x
        elapsed = self._now() - self.walk_started_at
        moved = abs(self.state.x - self.walk_start_x)
        arrived = abs(error) <= 5.0
        blocked = elapsed > 1.0 and moved < 4.0
        if arrived or blocked or elapsed > 4.0:
            self.keyboard.up(left_key)
            self.keyboard.up(right_key)
            target = self.walk_target
            self.pending = None
            self.walk_target = None
            if self.logger:
                self.logger.event(
                    "walk_done",
                    arrived=arrived,
                    blocked=blocked,
                    x=round(self.state.x, 1),
                    target=round(target, 1) if target is not None else None,
                )
            if self.verbose:
                print(
                    f"[walk] {'arrived' if arrived else 'blocked'} at x={self.state.x:.0f} "
                    f"(target {target:.0f})"
                )
            self.cooldown_until = self._now() + 0.15
            self.replan = True
            return
        if error > 0:
            self.keyboard.down(right_key)
            self.keyboard.up(left_key)
        else:
            self.keyboard.down(left_key)
            self.keyboard.up(right_key)
        if self.verbose and int(elapsed * 4) != int((elapsed - 0.02) * 4):
            from .win import is_foreground

            print(
                f"[walk] t={elapsed:.1f} x={self.state.x:.0f} target={self.walk_target:.0f} "
                f"moved={moved:.1f} held={sorted(self.keyboard.held())} "
                f"focus={is_foreground(self.window.hwnd)} events={self.keyboard.events_sent}"
            )

    def _log_flight(self) -> None:
        plan = self.current_plan
        if plan is None or self.logger is None:
            return
        elapsed = self._now() - self.jump_started_at
        index = min(int(elapsed * 60.0), len(plan.points) - 1)
        point = plan.points[index] if plan.points else None
        self.logger.event(
            "flight",
            t=round(elapsed, 3),
            x=round(self.state.x, 1),
            y=round(self.state.y, 1),
            px=round(point.x, 1) if point else None,
            py=round(point.y, 1) if point else None,
            vx=round(self.state.vx, 0),
            vy=round(self.state.vy, 0),
            charge=plan.charge_s,
            direction=plan.direction,
        )

    def _flight_control(self) -> None:
        # The horizontal air velocity is locked at takeoff and cannot be changed
        # mid-air (verified in game). The only thing left to do is to stop walking
        # in the jump direction right before touchdown, otherwise the landing is
        # measured ~15px past the real one.
        plan = self.current_plan
        if plan is None or plan.landing is None:
            return
        if self.state.vy > 0 and self.state.y >= plan.landing[1] - 6.0:
            self.keyboard.up(self.cfg.input.right_key)
            self.keyboard.up(self.cfg.input.left_key)

    def _jump_failed(self) -> None:
        plan = self.current_plan
        takeoff_x = self.jump_takeoff_x
        self.keyboard.release_all()
        self.current_plan = None
        if plan is not None:
            self.recent_failures.append(
                (
                    round(takeoff_x, 1),
                    round(plan.charge_s, 2),
                    float(plan.direction),
                    self._now(),
                )
            )
        self.blocked_takeoffs.append((takeoff_x, self._now()))
        self.cooldown_until = self._now() + 0.25
        self.replan = True
        if self.verbose:
            print(f"[jump-failed] no liftoff at x={takeoff_x:.0f} (state x={self.state.x:.0f})")
        if self.logger:
            self.logger.event(
                "jump_failed",
                x=round(takeoff_x, 1),
                state_x=round(self.state.x, 1),
            )

    def _finish_jump(self) -> None:
        self.keyboard.release_all()
        plan = self.current_plan
        takeoff_x = self.jump_takeoff_x
        self.current_plan = None
        self.jump_liftoff_y = None
        self.cooldown_until = self._now() + 0.25
        if plan is not None and plan.landing is not None:
            miss_x = abs(self.state.x - plan.landing[0])
            miss_y = abs(self.state.y - plan.landing[1])
            self.last_landing_result = (
                plan.landing,
                (self.state.x, self.state.y),
                (miss_x, miss_y),
            )
            if miss_x > 14 or miss_y > 14:
                self.recent_failures.append(
                    (
                        round(takeoff_x, 1),
                        round(plan.charge_s, 2),
                        float(plan.direction),
                        self._now(),
                    )
                )
                if self.verbose:
                    print(
                        f"[miss] planned=({plan.landing[0]:.0f},{plan.landing[1]:.0f}) "
                        f"actual=({self.state.x:.0f},{self.state.y:.0f})"
                    )
                if self.logger:
                    self.logger.event(
                        "jump_missed",
                        planned=[round(plan.landing[0], 1), round(plan.landing[1], 1)],
                        actual=[round(self.state.x, 1), round(self.state.y, 1)],
                    )
        row = self.screen_cell[0] if self.screen_cell is not None else 0
        level_y = self.state.y - row * 360.0
        if level_y > self.best_level_y + 200.0:
            self.stats.falls += 1
            self.recent_falls.append(level_y)
            if self.logger:
                self.logger.event("fall", level_y=round(level_y, 1))
        if level_y < self.best_level_y:
            self.best_level_y = level_y
        self.stats.best_y = min(self.stats.best_y, level_y)
