"""Exact BodyComp stepping: movement, collision, slopes and special tiles.

Mirrors JumpKing.exe's BodyComp.Update, LevelManager.CheckCollision,
SlopeBlock.Intersects and the block-type modifiers (wind, water, sand, ice,
snow, quark, teleport) as decompiled from JumpKing.exe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .gamephysics import (
    GRAVITY,
    JUMP_SPEED,
    MAX_FALL,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    SCREEN_W,
    SPEED,
    WALK_SPEED,
    WorldGrid,
    charge_intensity,
)
from .slopes import CollisionInfo, collision_info, slope_type_at  # noqa: F401

ICE_FRICTION = 0.1
WATER_MULTIPLIER = 0.5
SAND_VY_CAP = 0.75

__all__ = [
    "Body",
    "CollisionInfo",
    "StepResult",
    "StepEnv",
    "body_step",
    "apply_gravity",
    "simulate_jump",
    "simulate_fall",
    "simulate_walk",
    "start_jump",
    "hitbox_collides",
    "collision_info",
]


def _move_towards(value: float, target: float, step: float) -> float:
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


@dataclass
class Body:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    knocked: bool = False
    on_snow: bool = False
    on_ice: bool = False
    on_sand: bool = False
    wind_enabled: bool = False
    last_screen: int = -1

    def copy(self) -> Body:
        return Body(
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.on_ground,
            self.knocked,
            self.on_snow,
            self.on_ice,
            self.on_sand,
            self.wind_enabled,
            self.last_screen,
        )


@dataclass
class StepEnv:
    """Optional per-frame environment: the time-varying wind (px/frame added
    to velocity.X each airborne frame in wind screens) as a function of the
    run time in seconds."""

    wind: Callable[[float], float] | None = None
    start_time: float = 0.0
    fps: float = 60.0


@dataclass
class StepResult:
    outcome: str = "air"
    x: float = 0.0
    y: float = 0.0
    surface_y: float | None = None


def update_wind(body: Body, grid: WorldGrid, env: StepEnv, t: float) -> None:
    """BodyComp.UpdateWind + the p_wind gate."""
    if env.wind is None:
        return
    screen = -int(np.floor((body.y + 13.0) / 360.0))
    windy = screen in grid.wind_screens
    if not windy:
        body.wind_enabled = False
    if windy and body.on_ground:
        body.wind_enabled = True
    if body.last_screen > screen and windy:
        body.wind_enabled = True
    body.last_screen = screen
    if not body.on_snow and body.wind_enabled and not grid.has_nowind(body.x, body.y):
        body.vx += env.wind(t)


def body_step(body: Body, grid: WorldGrid) -> StepResult:
    """One exact BodyComp.Update step (movement + collision, no gravity)."""
    info = collision_info(grid, body.x, body.y)
    if info.hit:
        body.x += body.vx
        body.y += body.vy
        return StepResult("stuck", body.x, body.y)
    body.on_sand = info.sand
    last_vx, last_vy = body.vx, body.vy
    in_water = grid.is_in_water(body.x, body.y)
    mult = WATER_MULTIPLIER if in_water else 1.0
    # X move (sand moves at a quarter speed)
    direction_x = 1 if body.vx > 0 else -1
    moved_x = body.vx * mult
    if body.on_sand:
        moved_x *= 0.25
    body.x += moved_x
    if not grid.can_teleport_at(body.x + PLAYER_WIDTH / 2.0, body.y):
        body.x = max(-PLAYER_WIDTH / 2, min(SCREEN_W - PLAYER_WIDTH / 2, body.x))
    else:
        body.x, body.y = grid.teleport_wrap(body.x, body.y)
    info = collision_info(grid, body.x, body.y)
    flag2 = False
    if info.hit:
        body.x = float(int(body.x))
        info = collision_info(grid, body.x, body.y)
        while info.hit:
            body.x -= direction_x
            next_info = collision_info(grid, body.x, body.y)
            if not next_info.hit:
                break
            info = next_info
        if info.slope == 0:
            body.vx *= -0.5
            body.knocked = True
        else:
            flag2 = True
            if body.on_ground and abs(body.vx) <= WALK_SPEED and body.vy > 0:
                body.vx = 0.0
            elif last_vx * info.normal[0] + last_vy * info.normal[1] < 0:
                body.vx, body.vy = tangent_component(body.vx, body.vy, info.normal)
    # Y move
    direction_y = 1 if body.vy > 0 else -1
    if body.on_sand:
        if body.vy > 0:
            body.y += body.vy * mult
        else:
            body.y += body.vy * mult * 0.5
    else:
        body.y += body.vy * mult
        if body.on_ground and body.vy > 0:
            body.y += 1.0
    flag3 = False
    body.on_snow = False
    info = collision_info(grid, body.x, body.y)
    if info.hit:
        flag3 = True
        body.on_ice = info.ice
        body.y = float(int(body.y))
        info = collision_info(grid, body.x, body.y)
        while info.hit:
            body.y -= direction_y
            next_info = collision_info(grid, body.x, body.y)
            if not next_info.hit:
                break
            info = next_info
        if direction_y > 0:
            slope_hit = info.slope != 0
            body.on_ground = True
            if slope_hit:
                body.on_ground = False
                if last_vx * info.normal[0] + last_vy * info.normal[1] < 0:
                    body.vx, body.vy = tangent_component(body.vx, body.vy, info.normal)
                    body.knocked = True
                else:
                    slope_hit = False
            if not slope_hit:
                body.knocked = False
                if info.snow:
                    body.on_snow = True
            if info.slope == 0 or body.vy <= 0 or body.on_ground:
                body.vy = 0.0
            body.on_sand = info.sand
            if not slope_hit:
                return StepResult("landed", body.x, body.y, body.y + PLAYER_HEIGHT)
            return StepResult("wall", body.x, body.y)
        if info.slope == 0:
            body.vx *= 0.5
        elif last_vx * info.normal[0] + last_vy * info.normal[1] < 0:
            body.vx, tangent_y = tangent_component(body.vx, body.vy, info.normal)
            body.vy = tangent_y
            body.vx *= 0.5
        if info.slope == 0 or body.vy <= 0 or body.on_ground:
            body.vy = 0.0
        body.on_sand = info.sand
        return StepResult("wall", body.x, body.y)
    body.on_ground = False
    body.on_ice = False
    if info.quark and body.vy == MAX_FALL:
        body.vx = 0.0
        body.y = float(np.ceil(body.y / 8.0) * 8.0)
    if flag2 and not flag3 and direction_y > 0 and not body.knocked:
        body.vx *= -0.5
        body.knocked = True
    body.on_sand = info.sand
    return StepResult("air", body.x, body.y)


def apply_gravity(body: Body, mult: float = 1.0) -> None:
    body.vy = min(body.vy + GRAVITY * mult, MAX_FALL)


def post_gravity(body: Body) -> None:
    """The tail of BodyComp.Update: sand speed cap and ice friction."""
    if body.on_sand:
        body.vy = min(SAND_VY_CAP, body.vy)
        body.knocked = False
    if body.on_ice:
        body.vx = _move_towards(body.vx, 0.0, ICE_FRICTION)


def start_jump(body: Body, direction: int, frames: int) -> bool:
    """The game's JumpState.DoJump: returns False when the jump is cancelled
    (snow with a too-small charge)."""
    intensity = charge_intensity(frames)
    if body.on_snow and intensity < 0.3:
        if intensity <= 0.2:
            return False
        intensity = 0.3
    body.vy = -JUMP_SPEED * intensity
    body.vx = float(max(-SPEED, min(SPEED, body.vx + float(direction) * SPEED)))
    body.on_ground = False
    return True


def tangent_component(vx: float, vy: float, normal: tuple[float, float]) -> tuple[float, float]:
    perpendicular = (normal[1], -normal[0])
    along = vx * perpendicular[0] + vy * perpendicular[1]
    return along * perpendicular[0], along * perpendicular[1]


@dataclass
class SimFrame:
    t: int
    x: float
    y: float
    vx: float
    vy: float
    on_ground: bool


@dataclass
class JumpResult:
    frames: list[SimFrame] = field(default_factory=list)
    outcome: str = "timeout"
    landing: tuple[float, float] | None = None
    max_rise: float = 0.0
    peak_y: float = 0.0


def hitbox_collides(grid: WorldGrid, x: float, y: float) -> bool:
    return collision_info(grid, x, y).hit


def simulate_jump(
    grid: WorldGrid,
    x: float,
    y: float,
    direction: int,
    frames: int,
    max_frames: int = 400,
    vx_extra: float = 0.0,
    on_ice: bool = False,
    env: StepEnv | None = None,
    on_snow: bool = False,
    wind_enabled: bool = False,
) -> JumpResult:
    """Exact jump simulation starting on the frame the velocity is applied.

    The snow state is taken from the ground under the feet when the caller
    does not pass it: the game's `JumpState.DoJump` cancels charges at or
    below 0.2 intensity and boosts 0.2–0.3 to 0.3, so a takeoff on snow can
    never be simulated as a short hop.
    """
    if not on_snow and not on_ice:
        on_snow = collision_info(grid, x, y + PLAYER_HEIGHT + 1.0).snow
    body = Body(
        x=x,
        y=y,
        vx=vx_extra,
        vy=0.0,
        on_ground=True,
        on_snow=on_snow,
        wind_enabled=wind_enabled,
    )
    if not start_jump(body, direction, frames):
        return JumpResult(outcome="cancelled", landing=(x, y))
    result = JumpResult()
    result.peak_y = y
    for step_index in range(max_frames):
        if env is not None and env.wind is not None:
            update_wind(body, grid, env, env.start_time + step_index / env.fps)
        step = body_step(body, grid)
        result.frames.append(SimFrame(step_index, body.x, body.y, body.vx, body.vy, body.on_ground))
        result.peak_y = min(result.peak_y, body.y)
        if step.outcome == "landed":
            result.outcome = "landed"
            result.landing = (body.x, step.surface_y)
            break
        if step.outcome == "stuck":
            result.outcome = "stuck"
            result.landing = (body.x, body.y)
            break
        # A jump that falls this far below its takeoff can never land on a
        # target within the planner's drop window; stopping early halves the
        # cost of the many rejected candidate simulations.
        if body.y > y + 220:
            result.outcome = "fell"
            result.landing = (body.x, body.y)
            break
        mult = WATER_MULTIPLIER if grid.is_in_water(body.x, body.y) else 1.0
        apply_gravity(body, mult)
        post_gravity(body)
    result.max_rise = y - result.peak_y
    return result


def simulate_fall(
    grid: WorldGrid,
    x: float,
    y: float,
    vx: float = 0.0,
    max_frames: int = 400,
    env: StepEnv | None = None,
    wind_enabled: bool = False,
) -> tuple[float, float] | None:
    """Free fall from (x, y) with no jump; returns the (x, surface_y) landing."""
    body = Body(x=x, y=y, vx=vx, vy=0.0, on_ground=False, wind_enabled=wind_enabled)
    for step_index in range(max_frames):
        if env is not None and env.wind is not None:
            update_wind(body, grid, env, env.start_time + step_index / env.fps)
        step = body_step(body, grid)
        if step.outcome == "landed":
            return body.x, step.surface_y
        if step.outcome == "stuck" or body.y > y + 400:
            return None
        mult = WATER_MULTIPLIER if grid.is_in_water(body.x, body.y) else 1.0
        apply_gravity(body, mult)
        post_gravity(body)
    return None


def simulate_walk(
    grid: WorldGrid,
    x: float,
    y: float,
    direction: int,
    max_frames: int = 1200,
) -> tuple[float, float, bool]:
    """Walk in `direction` from (x, y) until the run ends or the king falls.

    Snow blocks walking (the game zeroes the walk velocity); returns the
    current (x, y, on_ground) immediately in that case.
    """
    body = Body(x=x, y=y, vx=0.0, vy=0.0, on_ground=True)
    start_y = y
    for _ in range(max_frames):
        if body.on_ground:
            if body.on_snow or body.on_ice:
                return body.x, body.y, body.on_ground
            body.vx = float(direction) * WALK_SPEED
        step = body_step(body, grid)
        if step.outcome == "stuck":
            return body.x, body.y, True
        if not body.on_ground and body.y > start_y + 30.0:
            break
        if step.outcome == "landed" and body.y > start_y + 30.0:
            break
        mult = WATER_MULTIPLIER if grid.is_in_water(body.x, body.y) else 1.0
        apply_gravity(body, mult)
        post_gravity(body)
    return body.x, body.y, body.on_ground
