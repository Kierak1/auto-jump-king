"""Slope collision geometry: SAT overlap and the game's slope orientation.

Mirrors JumpKing.exe's SlopeBlock.Intersects and LevelManager.GetSlopeType.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gamephysics import (
    FAKE,
    ICE,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    QUARK,
    SAND,
    SLOPE,
    SNOW,
    SOLID,
    TILE,
    WorldGrid,
)

SQRT2 = float(np.sqrt(2.0))

NORMALS = {
    1: (-1.0 / SQRT2, -1.0 / SQRT2),  # TopLeft
    2: (1.0 / SQRT2, -1.0 / SQRT2),  # TopRight
    3: (-1.0 / SQRT2, 1.0 / SQRT2),  # BottomLeft
    4: (1.0 / SQRT2, 1.0 / SQRT2),  # BottomRight
}

SLOPE_CORNER = {
    1: ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),  # TopLeft: bottom-right triangle
    2: ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),  # TopRight: bottom-left triangle
    3: ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),  # BottomLeft: top-left triangle
    4: ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),  # BottomRight: top-right triangle
}


@dataclass
class CollisionInfo:
    hit: bool = False
    slope: int = 0
    normal: tuple[float, float] = (0.0, 0.0)
    ice: bool = False
    snow: bool = False
    sand: bool = False
    quark: bool = False


def slope_type_at(grid: WorldGrid, tx: int, ty: int) -> int:
    """The game's GetSlopeType: neighbour-based orientation."""
    cached = grid.slope_types.get((tx, ty))
    if cached is not None:
        return cached

    def edge(x: int, y: int) -> bool:
        if x < 0 or x >= 60:
            return True
        value = grid.tile_type(x, y)
        if value in (SOLID, SLOPE, ICE, SNOW, SAND):
            return True
        # LevelManager.IsBlockEdge treats the decorative FAKE tile as an edge.
        return value == FAKE

    result = 0
    if edge(tx + 1, ty) and edge(tx, ty + 1):
        result = 1  # TopLeft
    elif edge(tx - 1, ty) and edge(tx, ty + 1):
        result = 2  # TopRight
    elif edge(tx + 1, ty) and edge(tx, ty - 1):
        result = 3  # BottomLeft
    elif edge(tx - 1, ty) and edge(tx, ty - 1):
        result = 4  # BottomRight
    grid.slope_types[(tx, ty)] = result
    return result


def _edge_normals(
    shape: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    result = []
    for i in range(len(shape)):
        x0, y0 = shape[i]
        x1, y1 = shape[(i + 1) % len(shape)]
        result.append((-(y1 - y0), x1 - x0))
    return tuple(result)


TRIANGLE_AXES = {tuple(corners): _edge_normals(corners) for corners in SLOPE_CORNER.values()}
RECT_AXES = ((1.0, 0.0), (0.0, 1.0))


def _span(
    shape: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    axis: tuple[float, float],
) -> tuple[float, float]:
    ax, ay = axis
    iterator = iter(shape)
    px, py = next(iterator)
    low = high = ax * px + ay * py
    for px, py in iterator:
        value = ax * px + ay * py
        if value < low:
            low = value
        elif value > high:
            high = value
    return low, high


def triangle_rect_overlap(
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    base_x: float,
    base_y: float,
    rx: float,
    ry: float,
    rw: float,
    rh: float,
) -> bool:
    """SAT overlap between a slope's solid triangle and the hitbox rect.

    `corners` are tile-local (0..1) units; `base_x`/`base_y` are the world
    coordinates of the tile's top-left corner. The triangle's edge normals are
    constant per slope type, and the rectangle is axis-aligned, so only five
    axes ever need testing.
    """
    triangle = [(base_x + cx * 8.0, base_y + cy * 8.0) for cx, cy in corners]
    rectangle = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
    for axis in TRIANGLE_AXES[corners] + RECT_AXES:
        t0, t1 = _span(triangle, axis)
        r0, r1 = _span(rectangle, axis)
        if t1 < r0 or r1 < t0:
            return False
    return True


def box_overlap(base_x: float, base_y: float, x: float, y: float) -> bool:
    """Does the hitbox overlap the slope tile's bounding box at all."""
    return (
        base_x < x + PLAYER_WIDTH
        and x < base_x + TILE
        and base_y < y + PLAYER_HEIGHT
        and y < base_y + TILE
    )


def collision_info(grid: WorldGrid, x: float, y: float) -> CollisionInfo:
    """The game's CheckCollision: box/slope overlap plus block type info."""
    info = CollisionInfo()
    # Fast path: the hitbox is usually in open air; the integral image answers
    # "any colliding tile or quark" in O(1) without touching the tile loop.
    if not grid.rect_has(grid._solid_sums, x, y):
        return info
    ix = int(x)
    iy = int(y)
    tx0 = ix // TILE
    tx1 = (ix + PLAYER_WIDTH - 1) // TILE
    ty0 = iy // TILE
    ty1 = (iy + PLAYER_HEIGHT - 1) // TILE
    if tx1 < 0 or tx0 > 59:
        return info
    r0 = ty0 + grid.offset
    r1 = ty1 + grid.offset
    if r1 < 0 or r0 >= grid.tiles.shape[0]:
        return info
    tx0 = max(0, tx0)
    tx1 = min(59, tx1)
    r0 = max(0, r0)
    r1 = min(grid.tiles.shape[0] - 1, r1)
    center_x = x + PLAYER_WIDTH / 2.0
    slope_dist = float("inf")
    slope_top_y = float("inf")
    box_top_y = float("inf")
    tiles = grid.tiles
    for tx in range(tx0, tx1 + 1):
        for row in range(r0, r1 + 1):
            value = int(tiles[row, tx])
            if value == 0:
                continue
            ty = row - grid.offset
            if value == QUARK:
                info.quark = True
                continue
            if value in (SOLID, ICE, SNOW, SAND):
                info.hit = True
                if value == ICE:
                    info.ice = True
                elif value == SNOW:
                    info.snow = True
                elif value == SAND:
                    info.sand = True
                top = ty * TILE
                if top < box_top_y:
                    box_top_y = top
                continue
            if value != SLOPE:
                continue
            s_type = slope_type_at(grid, tx, ty)
            if s_type == 0:
                continue
            corners = SLOPE_CORNER[s_type]
            base_x = tx * TILE
            base_y = ty * TILE
            # Bounding-box first: it rejects most slopes without the SAT.
            if box_overlap(base_x, base_y, x, y) and triangle_rect_overlap(
                corners, base_x, base_y, x, y, float(PLAYER_WIDTH), float(PLAYER_HEIGHT)
            ):
                info.hit = True
                distance = abs((tx + 0.5) * TILE - center_x)
                if distance < slope_dist:
                    slope_dist = distance
                    info.slope = s_type
                    info.normal = NORMALS[s_type]
                if base_y < slope_top_y:
                    slope_top_y = base_y
    if info.slope in (1, 2) and box_top_y <= slope_top_y:
        info.slope = 0
        info.normal = (0.0, 0.0)
    return info


def hitbox_collides(grid: WorldGrid, x: float, y: float) -> bool:
    return collision_info(grid, x, y).hit
