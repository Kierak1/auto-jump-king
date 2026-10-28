from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_CHARGE_V0: list[tuple[float, float]] = [
    (0.1, 69.8),
    (0.2, 197.0),
    (0.3, 262.0),
    (0.35, 326.0),
    (0.4, 390.0),
    (0.5, 453.0),
    (0.55, 490.0),
    (0.6, 524.0),
]


class SolidQuery(Protocol):
    def is_solid(self, wx: float, wy: float) -> bool: ...

    def state(self, wx: float, wy: float) -> str: ...


@dataclass
class PhysicsParams:
    gravity: float = 886.4
    air_max_speed: float = 205.0
    walk_speed: float = 88.0
    max_charge_s: float = 0.62
    terminal_speed: float = 700.0
    charge_v0: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_CHARGE_V0))

    def v0_for_charge(self, charge_s: float) -> float:
        points = sorted(self.charge_v0)
        if not points:
            return 0.0
        if charge_s <= points[0][0]:
            return points[0][1] * max(0.0, charge_s) / points[0][0]
        if charge_s >= points[-1][0]:
            return points[-1][1]
        for (c0, v0), (c1, v1) in zip(points, points[1:], strict=False):
            if c0 <= charge_s <= c1:
                frac = (charge_s - c0) / (c1 - c0)
                return v0 + frac * (v1 - v0)
        return points[-1][1]

    def charge_for_v0(self, v0: float) -> float:
        points = sorted(self.charge_v0)
        if not points:
            return 0.0
        if v0 <= points[0][1]:
            return points[0][0] * max(0.0, v0) / points[0][1]
        if v0 >= points[-1][1]:
            return self.max_charge_s
        for (c0, v0a), (c1, v0b) in zip(points, points[1:], strict=False):
            if v0a <= v0 <= v0b:
                frac = (v0 - v0a) / (v0b - v0a)
                return c0 + frac * (c1 - c0)
        return self.max_charge_s


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class JumpPlan:
    direction: int
    charge_s: float
    v0: float
    points: list[TrajectoryPoint] = field(default_factory=list)
    landing: tuple[float, float] | None = None
    landing_known: bool = False
    outcome: str = "timeout"
    max_rise: float = 0.0
    drift: str = "hold"

    @property
    def height(self) -> float:
        return self.max_rise


def simulate_jump(
    x0: float,
    y0: float,
    direction: int,
    charge_s: float,
    params: PhysicsParams,
    solid: SolidQuery,
    dt: float = 1.0 / 60.0,
    max_time: float = 5.0,
    player_half_width: float = 16.0,
    player_height: float = 26.0,
    drift: str = "hold",
) -> JumpPlan:
    v0 = params.v0_for_charge(charge_s)
    plan = JumpPlan(direction=direction, charge_s=charge_s, v0=v0, drift=drift)
    vx = float(direction) * params.air_max_speed if drift == "hold" else 0.0
    vy = -v0
    x, y = x0, y0
    t = 0.0
    best_y = y0
    gravity = params.gravity
    terminal = params.terminal_speed
    is_solid = solid.is_solid
    state_of = solid.state
    while t < max_time:
        prev_x, prev_y = x, y
        vy = min(vy + gravity * dt, terminal)
        if drift == "late" and vy > 0:
            vx = float(direction) * params.air_max_speed
        x += vx * dt
        y += vy * dt
        t += dt
        best_y = min(best_y, y)
        plan.points.append(TrajectoryPoint(t, x, y, vx, vy))
        span = max(abs(x - prev_x), abs(y - prev_y))
        substeps = max(2, int(span / 4.0) + 1)
        hit: tuple[float, float] | None = None
        prev_state = state_of(prev_x, prev_y)
        for k in range(1, substeps + 1):
            fx = prev_x + (x - prev_x) * k / substeps
            fy = prev_y + (y - prev_y) * k / substeps
            cur_state = state_of(fx, fy)
            if cur_state == "solid" and prev_state != "solid":
                hit = (
                    prev_x + (x - prev_x) * (k - 1) / substeps,
                    prev_y + (y - prev_y) * (k - 1) / substeps,
                )
                break
            prev_state = cur_state
        if hit is not None:
            hx, hy = hit
            if vy > 0:
                surface_y = float((int(hy // 8) + 1) * 8)
                above = state_of(hx, surface_y - 4.0)
                if above == "solid":
                    plan.outcome = "wall"
                    plan.landing = (hx, hy)
                else:
                    plan.landing = (hx, surface_y)
                    plan.outcome = "landed"
                    plan.landing_known = True
            elif vy < 0:
                plan.outcome = "ceiling"
                plan.landing = (hx, hy)
            else:
                plan.outcome = "wall"
                plan.landing = (hx, hy)
            break
        head_y = y - player_height
        body_hit = False
        dy = 0.0
        while dy <= player_height:
            for dx in (-player_half_width, player_half_width):
                if is_solid(x + dx, y - dy):
                    body_hit = True
                    break
            if body_hit:
                break
            dy += 2.0
        if body_hit:
            plan.outcome = "wall"
            plan.landing = (prev_x, prev_y)
            break
        if vy < 0 and state_of(x, head_y) == "solid":
            plan.outcome = "ceiling"
            plan.landing = (prev_x, prev_y)
            break
        if y > y0 + 260:
            plan.outcome = "fell"
            plan.landing = (x, y)
            break
    plan.max_rise = y0 - best_y
    return plan
