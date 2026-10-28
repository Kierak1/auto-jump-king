"""Platform extraction and jump option generation over the world grid.

Every option is validated with the exact frame-by-frame simulator, so a
candidate that lands is guaranteed by physics to land there (when executed
with the same frame count and takeoff position).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gamephysics import (
    CHARGE_FRAMES,
    GRAVITY,
    JUMP_SPEED,
    SCREENS,
    SPEED,
    TILE,
    WorldGrid,
    charge_frames_for_rise,
    simulate_jump,
)

MIN_PLATFORM_WIDTH = 16.0
MAX_RISE = 160.0
MAX_REACH_X = 340.0


@dataclass(frozen=True)
class Platform:
    top_y: float
    x0: float
    x1: float
    screen: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def contains(self, x: float, margin: float = 0.0) -> bool:
        return self.x0 - margin <= x <= self.x1 + margin


def screen_index_for_world_y(world_y: float) -> int:
    return -int(np.floor((world_y - 13.0) / 360.0))


def extract_platforms(
    grid: WorldGrid,
    max_screen: int = SCREENS,
    min_width: float = MIN_PLATFORM_WIDTH,
) -> list[Platform]:
    """All walkable surfaces: a solid tile with the tile above free, grouped
    into horizontal runs (in world coordinates)."""
    solid = grid.solid
    above = np.zeros_like(solid)
    above[1:] = solid[:-1]
    surface = solid & ~above
    platforms: list[Platform] = []
    for row in range(surface.shape[0]):
        values = surface[row]
        if not values.any():
            continue
        ty = row - grid.offset
        screen = screen_index_for_world_y(ty * TILE)
        if screen > max_screen:
            continue
        padded = np.concatenate(([0], values.astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(padded))
        for start, end in zip(edges[::2], edges[1::2], strict=False):
            width = (end - start) * TILE
            if width < min_width:
                continue
            platforms.append(
                Platform(
                    top_y=float(ty * TILE),
                    x0=float(start * TILE),
                    x1=float(end * TILE),
                    screen=screen,
                )
            )
    return platforms


def platform_below(platforms: list[Platform], x: float, surface_y: float) -> Platform | None:
    best: Platform | None = None
    for platform in platforms:
        if abs(platform.top_y - surface_y) > 4.0:
            continue
        if platform.x0 - 17.0 <= x <= platform.x1 + 1.0:
            if best is None or platform.width > best.width:
                best = platform
    return best


@dataclass
class Option:
    takeoff_x: float
    direction: int
    frames: int
    target: Platform
    landing: tuple[float, float]
    walk: float
    gain: float
    margin: float
    robust: bool
    clearance: float = 0.0
    target_value: float = 0.0

    @property
    def key(self) -> tuple[float, int, int]:
        return (round(self.takeoff_x, 1), self.direction, self.frames)

    def describe(self) -> str:
        direction = {1: "right", -1: "left", 0: "up"}[self.direction]
        return (
            f"{direction} jump, charge {self.frames}/36, "
            f"landing at ({self.landing[0]:.0f},{self.landing[1]:.0f}) "
            f"on platform y={self.target.top_y:.0f} "
            f"(gain {self.gain:.0f}px, margin {self.margin:.0f}px, "
            f"corner {self.clearance:.0f}px)"
        )


def flight_frames(v0: float, rise_to: float) -> float:
    """Frames until the descending branch reaches `rise_to` above the takeoff
    (discrete stepping: y(t) = y0 - v0*t + g*t*(t-1)/2)."""
    b = GRAVITY + 2.0 * v0
    discriminant = b * b - 8.0 * GRAVITY * rise_to
    if discriminant < 0:
        return 0.0
    return (b + np.sqrt(discriminant)) / (2.0 * GRAVITY)


def surface_run(grid: WorldGrid, x0: float, y0: float) -> tuple[float, float]:
    """The contiguous solid run under the feet at (x0, y0)."""
    tx = int(x0 // TILE)
    ty = int((y0 + 26.0) // TILE)
    left = tx
    while grid.is_solid_tile(left - 1, ty):
        left -= 1
    right = tx
    while grid.is_solid_tile(right + 1, ty):
        right += 1
    return float(left * TILE), float((right + 1) * TILE)


def _simulate(grid: WorldGrid, takeoff_x: float, y0: float, direction: int, frames: int):
    return simulate_jump(grid, takeoff_x, y0, direction, frames)


TAKEOFF_WINDOW = (-12.0, -6.0, 0.0, 6.0, 12.0)


def _takeoff_candidates(
    ideal: float,
    run_x0: float,
    run_x1: float,
) -> list[float]:
    safe_x0 = run_x0 + 2.0
    safe_x1 = run_x1 - 3.0
    candidates = [ideal + d for d in TAKEOFF_WINDOW]
    if ideal < safe_x0:
        edge = safe_x0
        candidates.extend([edge, edge + 8.0, edge + 16.0])
    if ideal > safe_x1:
        edge = safe_x1
        candidates.extend([edge, edge - 8.0, edge - 16.0])
    candidates = [c for c in candidates if safe_x0 <= c <= safe_x1]
    return sorted(set(round(c, 1) for c in candidates))


def corner_clearance(frames, platform: Platform, direction: int) -> float:
    """Min distance from the feet to the platform's leading top corner during
    the approach (frames where the feet are still above the platform top).

    A small value means the arc skims the corner: a tiny execution error
    (charge jitter, walk overshoot) would clip the side and bounce.
    """
    cx = platform.x0 if direction >= 0 else platform.x1
    cy = platform.top_y
    best = float("inf")
    for frame in frames:
        fx = frame.x + 9.0
        fy = frame.y + 26.0
        if fy >= cy + 4.0:
            continue
        distance = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
        if distance < best:
            best = distance
    return best


MIN_CLEARANCE = 5.0


def options_for(
    grid: WorldGrid,
    platforms: list[Platform],
    x0: float,
    y0: float,
    walk_x0: float | None = None,
    walk_x1: float | None = None,
    include_lower: bool = False,
    max_options: int = 40,
    values: dict[int, float] | None = None,
) -> list[Option]:
    """All robust jump options from a standing position (x0, y0).

    With `values` (the global route values keyed by platform id) the options
    are sorted by the target's route value, so the best next jump follows the
    way to the top even when it gains no height (sideways hops).
    """
    options: list[Option] = []
    y_top = y0 - MAX_RISE
    current = platform_below(platforms, x0, y0 + 26.0)
    targets = [
        p
        for p in platforms
        if y_top - 4.0 <= p.top_y <= y0 + 30.0
        and abs(p.center_x - x0) < MAX_REACH_X
        and p is not current
    ]
    targets.sort(key=lambda p: p.top_y)
    run_x0, run_x1 = surface_run(grid, x0, y0)
    if walk_x0 is None:
        walk_x0 = run_x0
        walk_x1 = run_x1

    def target_value(target: Platform) -> float:
        if values is not None:
            return values.get(id(target), target.top_y)
        return target.top_y

    for target in targets:
        rise = y0 - target.top_y
        f_min = max(1, charge_frames_for_rise(rise + 2.0))
        target_cx = min(max(target.center_x, target.x0 + 5.0), target.x1 - 5.0)
        for direction in (-1, 0, 1):
            if direction == 0:
                if not target.contains(x0, margin=1.0):
                    continue
                takeoff_sets = {f: [x0] for f in range(f_min, CHARGE_FRAMES + 1)}
            else:
                takeoff_sets = {}
                for frames in range(f_min, CHARGE_FRAMES + 1):
                    v0 = JUMP_SPEED * frames / CHARGE_FRAMES
                    t_flight = flight_frames(v0, rise)
                    if t_flight <= 0:
                        continue
                    ideal = target_cx - direction * SPEED * t_flight
                    if ideal < walk_x0 - 24.0 or ideal > walk_x1 + 24.0:
                        continue
                    takeoff_sets[frames] = _takeoff_candidates(ideal, walk_x0, walk_x1)
            for frames, takeoffs in takeoff_sets.items():
                best: Option | None = None
                for takeoff_x in takeoffs:
                    if walk_x0 is not None and not (walk_x0 <= takeoff_x <= walk_x1):
                        continue
                    if not grid.is_solid(takeoff_x, y0 + 26.0):
                        continue
                    result = _simulate(grid, takeoff_x, y0, direction, frames)
                    if result.outcome != "landed" or result.landing is None:
                        continue
                    lx, ly = result.landing
                    if abs(ly - target.top_y) > 4.0:
                        continue
                    if not target.contains(lx, margin=1.0):
                        continue
                    margin = min(lx - target.x0, target.x1 - lx)
                    if margin < 3.0:
                        continue
                    clearance = corner_clearance(result.frames, target, direction)
                    if clearance < MIN_CLEARANCE:
                        continue
                    if best is None or clearance > best.clearance + 0.5:
                        best = Option(
                            takeoff_x=takeoff_x,
                            direction=direction,
                            frames=frames,
                            target=target,
                            landing=(lx, ly),
                            walk=abs(takeoff_x - x0),
                            gain=rise,
                            margin=margin,
                            robust=False,
                            clearance=clearance,
                            target_value=target_value(target),
                        )
                if best is None:
                    continue
                refined = best
                for delta in (-2.0, 2.0):
                    candidate_x = best.takeoff_x + delta
                    if walk_x0 is not None and not (walk_x0 <= candidate_x <= walk_x1):
                        continue
                    result = _simulate(grid, candidate_x, y0, direction, frames)
                    if result.outcome != "landed" or result.landing is None:
                        continue
                    lx, ly = result.landing
                    if abs(ly - target.top_y) > 4.0 or not target.contains(lx, margin=1.0):
                        continue
                    margin = min(lx - target.x0, target.x1 - lx)
                    clearance = corner_clearance(result.frames, target, direction)
                    if margin < 3.0 or clearance < MIN_CLEARANCE:
                        continue
                    if (margin + clearance) > (refined.margin + refined.clearance):
                        refined = Option(
                            takeoff_x=candidate_x,
                            direction=direction,
                            frames=frames,
                            target=target,
                            landing=(lx, ly),
                            walk=abs(candidate_x - x0),
                            gain=rise,
                            margin=margin,
                            robust=False,
                            clearance=clearance,
                            target_value=target_value(target),
                        )
                robust = True
                if frames > 1:
                    lower = _simulate(grid, refined.takeoff_x, y0, direction, frames - 1)
                    robust &= (
                        lower.outcome == "landed"
                        and lower.landing is not None
                        and target.contains(lower.landing[0], margin=1.0)
                        and abs(lower.landing[1] - target.top_y) <= 4.0
                        and corner_clearance(lower.frames, target, direction) >= MIN_CLEARANCE
                    )
                if frames < CHARGE_FRAMES:
                    higher = _simulate(grid, refined.takeoff_x, y0, direction, frames + 1)
                    robust &= (
                        higher.outcome == "landed"
                        and higher.landing is not None
                        and target.contains(higher.landing[0], margin=1.0)
                        and abs(higher.landing[1] - target.top_y) <= 4.0
                        and corner_clearance(higher.frames, target, direction) >= MIN_CLEARANCE
                    )
                refined.robust = robust
                options.append(refined)
    if not options and include_lower:
        options.extend(_lower_options(grid, platforms, x0, y0, walk_x0, walk_x1))
    options.sort(
        key=lambda o: (
            o.target_value,
            o.target.top_y,
            not o.robust,
            -o.clearance,
            -o.margin,
            o.walk,
        )
    )
    deduped: list[Option] = []
    seen: set[tuple[float, int, int]] = set()
    for option in options:
        if option.key in seen:
            continue
        seen.add(option.key)
        deduped.append(option)
        if len(deduped) >= max_options:
            break
    return deduped


def _lower_options(
    grid: WorldGrid,
    platforms: list[Platform],
    x0: float,
    y0: float,
    walk_x0: float | None,
    walk_x1: float | None,
) -> list[Option]:
    """When nothing climbs, offer safe jumps down (rarely needed)."""
    options: list[Option] = []
    targets = [
        p for p in platforms if y0 + 30.0 <= p.top_y <= y0 + 200.0 and abs(p.center_x - x0) < 260
    ]
    for target in targets:
        for direction in (-1, 0, 1):
            for frames in range(2, CHARGE_FRAMES + 1, 2):
                v0 = JUMP_SPEED * frames / CHARGE_FRAMES
                t_flight = flight_frames(v0, y0 - target.top_y)
                takeoff_x = target.center_x - direction * SPEED * t_flight
                if walk_x0 is not None and not (walk_x0 <= takeoff_x <= walk_x1):
                    continue
                result = _simulate(grid, takeoff_x, y0, direction, frames)
                if result.outcome != "landed" or result.landing is None:
                    continue
                lx, ly = result.landing
                if not target.contains(lx, margin=2.0) or abs(ly - target.top_y) > 4.0:
                    continue
                margin = min(lx - target.x0, target.x1 - lx)
                clearance = corner_clearance(result.frames, target, direction)
                options.append(
                    Option(
                        takeoff_x=takeoff_x,
                        direction=direction,
                        frames=frames,
                        target=target,
                        landing=(lx, ly),
                        walk=abs(takeoff_x - x0),
                        gain=y0 - target.top_y,
                        margin=margin,
                        robust=False,
                        clearance=clearance,
                    )
                )
    return options
