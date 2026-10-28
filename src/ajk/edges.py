"""Backward jump-edge generation: all verified jumps that LAND on a target.

This is the core of the 4->3->2->1 planner: instead of asking "where can I
jump from here", we ask "which takeoffs land on this platform" and rely on the
exact frame simulator for proof. Edges are cached per (source, target) pair;
the tower is static and the physics deterministic, so a computed edge stays
valid for the session. Wind-aware plans use a cache epoch so time-varying wind
never reuses a stale arc.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from .gamephysics import JUMP_SPEED, PLAYER_WIDTH, SPEED, WorldGrid, charge_frames_for_rise
from .planner import Platform, corner_clearance, flight_frames
from .simulate import JumpResult, StepEnv, simulate_jump
from .slopes import collision_info

MAX_RISE = 158.0
MAX_DROP = 170.0
MIN_CLEARANCE = 3.0
MIN_CLEARANCE_WIND = 5.0
MIN_MARGIN = 3.0
MIN_MARGIN_WIND = 8.0
FRAME_SPAN = 8
TAKEOFF_OFFSETS = (0.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0, -12.0, -16.0)
SWEEP_STEP = 6.0
MAX_EDGES_PER_PAIR = 3
SUPPORT_MARGIN = 3.0
# Wind cache epochs are phase buckets; keys at or above the base are dropped
# on load so wind arcs always recompute with the current filters.
WIND_EPOCH_BASE = 1_000_000


@dataclass
class Edge:
    source: int
    target: int
    takeoff_x: float
    direction: int
    frames: int
    landing: tuple[float, float]
    robust: bool
    clearance: float
    margin: float
    gain: float
    support: float = 18.0
    bounce: bool = False

    def describe(self) -> str:
        direction = {1: "right", -1: "left", 0: "up"}[self.direction]
        return (
            f"{direction} jump charge {self.frames}/36 from x={self.takeoff_x:.0f} "
            f"-> ({self.landing[0]:.0f},{self.landing[1]:.0f}) "
            f"gain {self.gain:.0f}px margin {self.margin:.0f}px corner {self.clearance:.0f}px"
        )


CACHE_VERSION = 8


class EdgeFactory:
    """Verified incoming edges for platforms, with a session cache.

    The level is static and the physics deterministic, so the cache can be
    persisted between runs (`cache_path`); it is keyed by a hash of the
    platform geometry so a changed level extraction invalidates it.
    """

    def __init__(
        self, grid: WorldGrid, platforms: list[Platform], cache_path: Path | None = None
    ) -> None:
        self.grid = grid
        self.platforms = platforms
        self.env: StepEnv | None = None
        self.wind_enabled = False
        self.epoch = 0
        self._cache: dict[tuple[int, int, int], list[Edge]] = {}
        self._standable: dict[int, list[float]] = {}
        # Hard wall-clock limit for expensive edge computations; an aborted
        # pair is left uncached so a later session can finish it.
        self.deadline: float | None = None
        self._cache_path = cache_path
        self._geometry = hashlib.sha1(repr(platforms).encode("utf-8")).hexdigest()
        self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with self._cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("version") != CACHE_VERSION or payload.get("geometry") != self._geometry:
            return
        edges = payload.get("edges")
        if isinstance(edges, dict):
            self._cache = {key: value for key, value in edges.items() if key[0] < WIND_EPOCH_BASE}

    def save_cache(self) -> None:
        if self._cache_path is None:
            return
        # Wind epochs are phase buckets within the wind period (bounded key
        # space), so they persist too and planning stays warm across runs.
        edges = self._cache
        payload = {"version": CACHE_VERSION, "geometry": self._geometry, "edges": edges}
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(self._cache_path)
        except OSError:
            return

    def standable(self, source_index: int) -> list[float]:
        """Takeoff x positions on a platform where the hitbox is neither inside
        a wall nor floating. The support must also hold `SUPPORT_MARGIN` px to
        both sides of the position: a takeoff hanging on a 1-2px corner sliver
        cannot be walked to (the walker would fall off the edge)."""
        cached = self._standable.get(source_index)
        if cached is not None:
            return cached
        source = self.platforms[source_index]
        y = source.top_y - 26.0
        xs: list[float] = []
        x = source.x0 - 16.0
        while x <= source.x1:
            free = not collision_info(self.grid, x, y).hit
            supported = (
                collision_info(self.grid, x - SUPPORT_MARGIN, y + 1.0).hit
                and collision_info(self.grid, x + SUPPORT_MARGIN, y + 1.0).hit
            )
            if free and supported:
                xs.append(round(x, 1))
            x += 2.0
        self._standable[source_index] = xs
        return xs

    @staticmethod
    def _snap(xs: list[float], x: float) -> float:
        index = bisect_left(xs, x)
        if index >= len(xs):
            return xs[-1]
        if index == 0:
            return xs[0]
        left, right = xs[index - 1], xs[index]
        return left if x - left <= right - x else right

    def set_wind(self, env: StepEnv | None, wind_enabled: bool, epoch: int) -> None:
        self.env = env
        self.wind_enabled = wind_enabled
        self.epoch = epoch

    def simulate(self, x: float, y: float, direction: int, frames: int) -> JumpResult:
        """Wind-aware jump simulation with the factory's current environment."""
        return simulate_jump(
            self.grid,
            x,
            y,
            direction,
            frames,
            env=self.env,
            wind_enabled=self.wind_enabled,
        )

    def _arc_has_clearance(self, result: JumpResult, sides: bool = True) -> bool:
        """Reject arcs whose simulated hitbox overlaps solid tiles.

        The simulator resolves collisions exactly, so this only catches tiles
        the stepping may have skipped; designed routes legitimately squeeze
        through a zero-pixel gap, which is safe because the physics is
        frame-exact and every edge is re-verified from the live takeoff.
        `sides=False` (wall bounces) checks the head row only, where the wall
        legitimately runs along the body."""
        is_solid = self.grid.is_solid
        for frame in result.frames:
            if frame.on_ground:
                continue
            ix = int(frame.x)
            iy = int(frame.y)
            if sides:
                points = [
                    (ix, iy),
                    (ix + PLAYER_WIDTH // 2, iy),
                    (ix + PLAYER_WIDTH - 1, iy),
                    (ix, iy + 6),
                    (ix + PLAYER_WIDTH - 1, iy + 6),
                ]
            else:
                points = [
                    (ix + 1, iy),
                    (ix + PLAYER_WIDTH // 2, iy),
                    (ix + PLAYER_WIDTH - 1, iy),
                ]
            for x, y in points:
                if is_solid(x, y):
                    return False
        return True

    @staticmethod
    def _no_ceiling_rebound(result: JumpResult) -> bool:
        """Ceiling rebounds are unpredictable in the live game; wall bounces
        are allowed only when proven stable (see `_wall_bounce`)."""
        for prev, cur in zip(result.frames, result.frames[1:], strict=False):
            if prev.vy > 0.5 and cur.vy < -0.5:
                return False
        return True

    def _wall_bounce(self, result: JumpResult, direction: int) -> bool:
        if self.env is not None and self.env.wind is not None:
            return False
        if direction != 0:
            return any(frame.vx * direction < -0.05 for frame in result.frames)
        return any(abs(frame.vx) > 0.05 for frame in result.frames)

    @staticmethod
    def _lands_on(result: JumpResult, target: Platform) -> bool:
        """The body must end up standing with part of the hitbox on the target:
        the hitbox left edge lands at most 17px left of the platform start and
        the body must overlap it by at least MIN_MARGIN px on both cases."""
        if result.outcome != "landed" or result.landing is None:
            return False
        lx, ly = result.landing
        if abs(ly - target.top_y) > 2.0:
            return False
        if lx + PLAYER_WIDTH <= target.x0 + 1.0:
            return False
        if lx >= target.x1 - 1.0:
            return False
        return True

    @staticmethod
    def _margin(result: JumpResult, target: Platform) -> float:
        lx = result.landing[0]
        return min(lx + PLAYER_WIDTH - target.x0, target.x1 - lx)

    def _ready(self, target: Platform, source: Platform) -> bool:
        rise = source.top_y - target.top_y
        if rise > MAX_RISE or rise < -MAX_DROP:
            return False
        # Horizontal reachability: a full charge plus air drift bounds how far
        # the landing can be from any takeoff on the source platform.
        flight = flight_frames(JUMP_SPEED, max(rise, 0.0))
        if flight <= 0.0:
            return False
        gap = max(source.x0 - target.x1, target.x0 - source.x1, 0.0)
        return gap <= SPEED * flight + 40.0

    def edges_to(self, target_index: int, source_index: int) -> list[Edge]:
        key = (self.epoch, source_index, target_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        target = self.platforms[target_index]
        source = self.platforms[source_index]
        edges = self._compute(target_index, source_index) if self._ready(target, source) else []
        if edges is None:
            return []
        self._cache[key] = edges
        return edges

    def _candidates(
        self,
        target: Platform,
        source: Platform,
        frames_list: list[int],
        rise: float,
        standable: list[float],
    ) -> list[tuple[int, int, float]]:
        safe0 = standable[0]
        safe1 = standable[-1]
        out: list[tuple[int, int, float]] = []
        for direction in (-1, 0, 1):
            aims: list[float] = []
            frames_here = set(frames_list)
            if direction == 0:
                if target.x0 <= source.center_x <= target.x1:
                    aims = [min(max(source.center_x, safe0), safe1)]
            else:
                near = target.x0 + 6.0 if direction > 0 else target.x1 - 6.0
                deep = target.x0 + 24.0 if direction > 0 else target.x1 - 24.0
                aims = [target.center_x, near, deep]
                gap = (near - safe1) if direction > 0 else (safe0 - near)
                if gap > 0.0:
                    needed = max(1, int(round(gap / 6.61)))
                    frames_here.update(f for f in range(needed - 3, needed + 4) if 1 <= f <= 36)
                frames_here.update(range(max(1, min(frames_list)), 37, 3))
            for frames in sorted(frames_here):
                v0 = JUMP_SPEED * frames / 36.0
                t_flight = flight_frames(v0, rise)
                if t_flight <= 0.0 and direction != 0:
                    continue
                for aim in aims:
                    if direction == 0:
                        out.append((direction, frames, self._snap(standable, aim)))
                        continue
                    ideal = aim - direction * 3.5 * t_flight
                    for off in TAKEOFF_OFFSETS:
                        out.append((direction, frames, self._snap(standable, ideal + off)))
                    if ideal < safe0 or ideal > safe1:
                        edge_x = safe0 if ideal < safe0 else safe1
                        out.append((direction, frames, edge_x))
                        back = -6.0 if ideal > safe1 else 6.0
                        out.append((direction, frames, self._snap(standable, edge_x + back)))
                        out.append((direction, frames, self._snap(standable, edge_x + 2 * back)))
            sweep_frames = {min(frames_list) + 2, max(frames_list), max(frames_here)}
            for frames in sweep_frames:
                x = safe0
                while x <= safe1:
                    out.append((direction, frames, self._snap(standable, x)))
                    x += SWEEP_STEP
        seen: set[tuple[int, int, float]] = set()
        ordered: list[tuple[int, int, float]] = []
        for direction, frames, x in out:
            key = (direction, frames, x)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    @staticmethod
    def _collision_kick(result: JumpResult) -> bool:
        """A wall or slope deflected the arc: velocity.X changes by far more
        than the wind ever adds in one frame (<= 0.1 px/frame). Such arcs are
        chaotic once wind is involved."""
        for a, b in zip(result.frames, result.frames[1:], strict=False):
            if abs(b.vx - a.vx) > 0.25:
                return True
        return False

    def _compute(self, target_index: int, source_index: int) -> list[Edge] | None:
        """Edges landing on `target` from `source`; None when the wall-clock
        deadline hit before the pair could be finished (result not cached)."""
        target = self.platforms[target_index]
        source = self.platforms[source_index]
        standable = self.standable(source_index)
        if not standable:
            return []
        rise = source.top_y - target.top_y
        y0 = source.top_y - 26.0
        f_min = max(1, charge_frames_for_rise(max(2.0, rise + 2.0)))
        low = max(1, f_min - 2)
        frames_list = sorted({min(36, f) for f in range(low, f_min + FRAME_SPAN)})
        frames_list.sort(key=lambda f: abs(f - (f_min + 2)))
        found: list[Edge] = []
        candidates = self._candidates(target, source, frames_list, rise, standable)
        for direction, frames, takeoff_x in candidates:
            if self.deadline is not None and time.monotonic() > self.deadline:
                return None
            result = self.simulate(takeoff_x, y0, direction, frames)
            if result.outcome != "landed" or not self._lands_on(result, target):
                continue
            if not self._no_ceiling_rebound(result):
                continue
            if self.env is not None and self.env.wind is not None and self._collision_kick(result):
                # In wind, wall/slope deflections cannot be replayed reliably:
                # the phase shifts the graze and the arc goes anywhere.
                continue
            bouncy = self._wall_bounce(result, direction)
            if not self._arc_has_clearance(result, sides=not bouncy):
                continue
            if bouncy and not self._bounce_stable(target, y0, direction, frames, takeoff_x):
                continue
            margin = self._margin(result, target)
            wind = self.env is not None and self.env.wind is not None
            if margin < (MIN_MARGIN_WIND if wind else MIN_MARGIN):
                continue
            clearance = corner_clearance(result.frames, target, direction)
            min_clearance = (
                MIN_CLEARANCE_WIND
                if self.env is not None and self.env.wind is not None
                else MIN_CLEARANCE
            )
            if clearance < min_clearance and not bouncy:
                continue
            support = min(takeoff_x + PLAYER_WIDTH, source.x1) - max(takeoff_x, source.x0)
            found.append(
                Edge(
                    source=source_index,
                    target=target_index,
                    takeoff_x=takeoff_x,
                    direction=direction,
                    frames=frames,
                    landing=result.landing,
                    robust=False,
                    clearance=clearance,
                    margin=margin,
                    gain=-rise,
                    support=support,
                    bounce=bouncy,
                )
            )
            if len(found) >= 6:
                break
        if not found:
            return []
        found.sort(
            key=lambda e: (
                e.bounce,
                -(e.clearance + e.margin) + (0.0 if e.support >= 5.0 else 25.0),
                abs(e.takeoff_x - source.center_x),
            )
        )
        kept: list[Edge] = []
        used: list[tuple[float, int]] = []
        for edge in found:
            key = (edge.takeoff_x, edge.direction)
            if any(abs(edge.takeoff_x - u) < 1.0 and edge.direction == d for u, d in used):
                continue
            used.append(key)
            kept.append(edge)
            if len(kept) >= MAX_EDGES_PER_PAIR:
                break
        kept[0].robust = self._robust(kept[0])
        return kept

    def _bounce_stable(
        self,
        target: Platform,
        y0: float,
        direction: int,
        frames: int,
        takeoff_x: float,
    ) -> bool:
        """A designed wall bounce is trusted after the nominal simulation plus
        the execution-time verify (which re-simulates from the exact takeoff).
        The charge is frame-locked and compensated for the game's extra sample
        frame, so no charge perturbation is modelled here."""
        result = self.simulate(takeoff_x, y0, direction, frames)
        if not self._lands_on(result, target):
            return False
        if not self._no_ceiling_rebound(result):
            return False
        if self._margin(result, target) < MIN_MARGIN:
            return False
        return True

    def _robust(self, edge: Edge) -> bool:
        target = self.platforms[edge.target]
        source = self.platforms[edge.source]
        y0 = source.top_y - 26.0
        for frames in (edge.frames - 1, edge.frames + 1):
            if frames < 1 or frames > 36:
                continue
            result = self.simulate(edge.takeoff_x, y0, edge.direction, frames)
            if not self._lands_on(result, target):
                return False
            if not self._no_ceiling_rebound(result):
                return False
            if not self._arc_has_clearance(result):
                return False
            if corner_clearance(result.frames, target, edge.direction) < MIN_CLEARANCE:
                return False
        return True

    def verified(self, edge: Edge, takeoff_x: float, y0: float) -> JumpResult | None:
        """Re-check an edge from a concrete takeoff position; returns the
        simulated result when it is safe to execute, else None."""
        result = self.simulate(takeoff_x, y0, edge.direction, edge.frames)
        target = self.platforms[edge.target]
        wind = self.env is not None and self.env.wind is not None
        if not self._lands_on(result, target):
            return None
        if not self._no_ceiling_rebound(result):
            return None
        if not self._arc_has_clearance(result, sides=not edge.bounce):
            return None
        if self._margin(result, target) < (MIN_MARGIN_WIND if wind else MIN_MARGIN):
            return None
        if wind and not edge.bounce and self._collision_kick(result):
            # The live phase put the arc into a wall/slope; its landing is not
            # reproducible, so never execute it.
            return None
        if (
            wind
            and not edge.bounce
            and corner_clearance(result.frames, target, edge.direction) < MIN_CLEARANCE_WIND
        ):
            # A wind phase shift moves the arc up to tens of px; a corner
            # graze would flip into a wall bounce mid-execution.
            return None
        return result

    def verify(self, edge: Edge, takeoff_x: float, y0: float) -> bool:
        """Re-check an edge from a concrete takeoff position before executing."""
        return self.verified(edge, takeoff_x, y0) is not None
