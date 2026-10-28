"""Exact Jump King physics and collision, derived from the decompiled game.

Facts (JumpKing.exe, `PlayerValues`, `BodyComp`, `JumpState`, `LevelManager`):

- The game runs a fixed 60 Hz update. All physics is per-frame, not per-second.
- GRAVITY  = JUMP_HEIGHT / CalculateJumpHeight(34, 1) = 153 / 595 = 0.2571428 px/frame^2
- JUMP     = 34 * GRAVITY = 8.742857 px/frame  (full charge)
- charge intensity is linear in held frames: min(1, frames / 36); velocity.Y = JUMP * intensity
- air drift SPEED = 3.5 px/frame, walk speed 1.5 px/frame, MAX_FALL = 10 px/frame
- hitbox is 18 x 26 with the integer top-left corner at (int)pos
- per frame: move X, resolve; move Y, resolve; then apply gravity
- the level is one vertical strip of 169 screens (60x45 tiles of 8px each);
  screen index s occupies world y in [-s*360, -s*360 + 360)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .level import _decode_texture

FPS = 60
PLAYER_WIDTH = 18
PLAYER_HEIGHT = 26
GRAVITY = 153.0 / 595.0
JUMP_SPEED = 34.0 * GRAVITY
SPEED = 3.5
WALK_SPEED = 1.5
MAX_FALL = 10.0
JUMP_TIME = 0.6
CHARGE_FRAMES = 36
SCREEN_W = 480
SCREEN_H = 360
TILE = 8
SCREENS = 169
SCREEN_ROWS = 13
SCREEN_COLS = 13
TILES_PER_SCREEN = 45
WORLD_TILE_MIN = -((SCREENS - 1) * TILES_PER_SCREEN)

EMPTY = 0
SOLID = 1
SLOPE = 2
ICE = 3
SNOW = 4
SAND = 5
WATER = 6
NOWIND = 7
QUARK = 8
TELEPORT = 9
WIND = 10
FAKE = 11

COLLIDING = (SOLID, SLOPE, ICE, SNOW, SAND)

WIND_OMEGA = 0.48124886
WIND_STRENGTH = 0.1


def wind_velocity(t_seconds: float) -> float:
    """WindManager.CurrentVelocityRaw: px/frame added to velocity.X per frame.

    `t_seconds` is the attempt timer (AchievementManager's current stats
    timeSpan), i.e. the game frame counter divided by 60.
    """
    x = t_seconds * WIND_OMEGA
    value = math.sin(x) * 2.0 + (1.0 if math.cos(x) > 0.0 else -1.0)
    return max(-1.0, min(1.0, value)) * WIND_STRENGTH


def charge_intensity(frames: int) -> float:
    return min(1.0, max(0, frames) / CHARGE_FRAMES)


def jump_velocity(frames: int) -> float:
    return JUMP_SPEED * charge_intensity(frames)


def charge_frames_for_rise(rise: float) -> int:
    """Smallest frame count whose jump apex clears `rise` pixels.

    Height is quadratic in the jump velocity (v0 is linear in the charge
    frames), so the required intensity scales with sqrt(rise / max_rise).
    """
    max_rise = JUMP_SPEED * JUMP_SPEED / (2.0 * GRAVITY)
    need = min(1.0, np.sqrt(max(0.0, rise) / max_rise))
    return int(np.ceil(need * CHARGE_FRAMES - 1e-9))


def screen_index_for_y(y: float) -> int:
    return -int(np.floor((y + 13.0) / 360.0))


def screen_origin_y(index: int) -> float:
    return -index * 360.0


class WorldGrid:
    """Whole-level collision, one byte per 8px tile, in strip coordinates.

    Tile (tx, ty) covers world pixels [tx*8, tx*8+8) x [ty*8, ty*8+8).
    Row `ty + GRID_OFFSET` of `tiles` is the row for world tile y = ty.

    The game samples the level texture one texel to the right of the natural
    block origin (`GetCoord` -> `num * TILE.X + x` against a texture whose
    per-screen content starts one texel later), measured live with floor-edge
    probes on multiple screens. We reproduce that by reading each screen block
    from texel `col * 60 + 1`.
    """

    TEXEL_SHIFT = 1

    @classmethod
    def _screen_block(cls, codes: np.ndarray, screen: int) -> np.ndarray:
        col, row = screen // SCREEN_ROWS, screen % SCREEN_ROWS
        x0 = col * 60 + cls.TEXEL_SHIFT
        block = codes[row * 45 : row * 45 + 45, x0 : x0 + 60]
        if block.shape[1] < 60:
            block = np.pad(block, ((0, 0), (0, 60 - block.shape[1])), constant_values=0)
        return block

    def __init__(self, game_dir: Path) -> None:
        img = _decode_texture(Path(game_dir) / "Content" / "level.xnb")
        b, g, r, a = (img[:, :, i] for i in range(4))
        codes = np.zeros(img.shape[:2], dtype=np.uint8)
        opaque = a > 0
        solid = opaque & (r == 0) & (g == 0) & (b == 0)
        slope = (r == 255) & (g == 0) & (b == 0)
        ice = (r == 0) & (g == 255) & (b == 255)
        snow = (r == 255) & (g == 255) & (b == 0)
        sand = (r == 255) & (g == 106) & (b == 0)
        water = (r == 0) & (g == 170) & (b == 170)
        nowind = (r == 255) & (g == 255) & (b == 255)
        quark = (r == 182) & (g == 255) & (b == 0)
        teleport = (g == 0) & (b == 255) & (r != 0)
        wind = (r == 0) & (g == 255) & (b == 0) & ~teleport
        fake = (r == 128) & (g == 128) & (b == 128)
        codes[solid] = SOLID
        codes[slope] = SLOPE
        codes[ice] = ICE
        codes[snow] = SNOW
        codes[sand] = SAND
        codes[water] = WATER
        codes[nowind] = NOWIND
        codes[quark] = QUARK
        codes[teleport] = TELEPORT
        codes[wind] = WIND
        codes[fake] = FAKE
        self.offset = -WORLD_TILE_MIN  # (SCREENS - 1) * TILES_PER_SCREEN
        self.tiles = np.zeros((SCREENS * TILES_PER_SCREEN, 60), dtype=np.uint8)
        for s in range(SCREENS):
            block = self._screen_block(codes, s)
            start = self.offset - s * TILES_PER_SCREEN
            self.tiles[start : start + 45] = block
        self.solid = np.isin(self.tiles, COLLIDING)
        # Integral images give O(1) "does this rect touch tile type X" queries,
        # which dominate the per-frame cost of the exact simulator.
        self._solid_sums = self._integral(self.solid | (self.tiles == QUARK))
        water = self.tiles == WATER
        nowind = self.tiles == NOWIND
        self._water_sums = self._integral(water)
        self._nowind_sums = self._integral(nowind)
        self.has_any_water = bool(water.any())
        self.has_any_nowind = bool(nowind.any())
        self._last_row = self.tiles.shape[0] - 1
        self.slope_types: dict[tuple[int, int], int] = {}
        self.teleport: dict[int, int] = {}
        teleport_codes = np.where(teleport, r, 0)
        for s in range(SCREENS):
            block = self._screen_block(teleport_codes, s)
            values = np.unique(block[block > 0])
            if values.size:
                self.teleport[s] = int(values[0])
        self.wind_screens = {
            s for s in range(SCREENS) if (self._screen_block(codes, s) == WIND).any()
        }
        self._nowind_rows: dict[int, np.ndarray] | None = None

    @staticmethod
    def _integral(mask: np.ndarray) -> np.ndarray:
        sums = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
        sums[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)
        return sums

    def rect_has(self, sums: np.ndarray, x: float, y: float) -> bool:
        """Does the integer hitbox at (x, y) overlap any True tile?

        Hot path: branchy and allocation-free on purpose.
        """
        ix = int(x)
        iy = int(y)
        tx0 = ix // TILE
        tx1 = (ix + PLAYER_WIDTH - 1) // TILE
        ty0 = iy // TILE
        ty1 = (iy + PLAYER_HEIGHT - 1) // TILE
        if tx1 < 0 or tx0 > 59:
            return False
        r0 = ty0 + self.offset
        r1 = ty1 + self.offset
        last = self._last_row
        if r1 < 0 or r0 > last:
            return False
        if r0 < 0:
            r0 = 0
        if r1 > last:
            r1 = last
        if tx0 < 0:
            tx0 = 0
        if tx1 > 59:
            tx1 = 59
        value = sums[r1 + 1, tx1 + 1] - sums[r0, tx1 + 1] - sums[r1 + 1, tx0] + sums[r0, tx0]
        return value > 0

    def tile_type(self, tx: int, ty: int) -> int:
        if tx < 0 or tx >= 60:
            return EMPTY
        row = ty + self.offset
        if row < 0 or row >= self.tiles.shape[0]:
            return EMPTY
        return int(self.tiles[row, tx])

    def is_solid_tile(self, tx: int, ty: int) -> bool:
        return self.tile_type(tx, ty) in COLLIDING

    def surface_below(self, x: float, y: float, limit: float = 400.0) -> float | None:
        """World y of the first solid surface below a point, if any."""
        tx = int(x // TILE)
        ty = int(y // TILE)
        rows = self.tiles[:, tx]
        max_row = min(rows.shape[0] - 1, ty + self.offset + int(limit // TILE))
        for row in range(ty + self.offset, max_row + 1):
            if int(rows[row]) in COLLIDING:
                return float((row - self.offset) * TILE)
        return None

    def type_at(self, x: float, y: float) -> int:
        return self.tile_type(int(np.floor(x / TILE)), int(np.floor(y / TILE)))

    def is_solid(self, x: float, y: float) -> bool:
        return self.type_at(x, y) in COLLIDING

    def solid_columns(self) -> np.ndarray:
        return np.isin(self.tiles, COLLIDING)

    def is_in_water(self, x: float, y: float) -> bool:
        if not self.has_any_water:
            return False
        return self.rect_has(self._water_sums, x, y)

    def has_nowind(self, x: float, y: float) -> bool:
        if not self.has_any_nowind:
            return False
        return self.rect_has(self._nowind_sums, x, y)

    @staticmethod
    def screen_for_y(y: float) -> int:
        return -int(np.floor((y + 13.0) / 360.0))

    def can_teleport_at(self, center_x: float, y: float) -> bool:
        screen = self.screen_for_y(y)
        if screen not in self.teleport:
            return False
        return center_x < 0.0 or center_x > 480.0

    def teleport_wrap(self, x: float, y: float) -> tuple[float, float]:
        """BodyComp.CapPosition on a teleport screen: wrap x and shift screens."""
        if x + PLAYER_WIDTH / 2.0 < 0.0:
            x += 480.0
        elif x + PLAYER_WIDTH / 2.0 > 480.0:
            x -= 480.0
        screen = self.screen_for_y(y)
        target = self.teleport.get(screen)
        if target is None:
            return x, y
        num = math.trunc((360.0 - y) / 360.0)
        num2 = (target - 1) - num
        return x, y - 360.0 * num2


def hitbox_collides(grid: WorldGrid, x: float, y: float) -> bool:
    from .simulate import hitbox_collides as impl

    return impl(grid, x, y)


def simulate_jump(*args, **kwargs):
    from .simulate import simulate_jump as impl

    return impl(*args, **kwargs)


def simulate_fall(*args, **kwargs):
    from .simulate import simulate_fall as impl

    return impl(*args, **kwargs)


def body_step(*args, **kwargs):
    from .simulate import body_step as impl

    return impl(*args, **kwargs)


def apply_gravity(*args, **kwargs):
    from .simulate import apply_gravity as impl

    return impl(*args, **kwargs)


from .simulate import Body, JumpResult, SimFrame, StepResult  # noqa: E402,F401
