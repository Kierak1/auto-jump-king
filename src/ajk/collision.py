from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .level import LevelAtlas

CELL = 8


@dataclass
class CollisionMap:
    cell: int = CELL
    solid_hits: dict[tuple[int, int], int] = field(default_factory=dict)
    free_hits: dict[tuple[int, int], int] = field(default_factory=dict)
    predicted_solid: set[tuple[int, int]] = field(default_factory=set)
    predicted_free: set[tuple[int, int]] = field(default_factory=set)
    atlas: LevelAtlas | None = None
    cell_row: int = 0
    cell_col: int = 0
    tile_set: set[tuple[int, int]] = field(default_factory=set)
    neighbor_tiles: set[tuple[int, int]] = field(default_factory=set)

    def _cell(self, wx: float, wy: float) -> tuple[int, int]:
        return int(np.floor(wx / self.cell)), int(np.floor(wy / self.cell))

    def set_cell(self, row: int, col: int, clear_learned: bool = True) -> None:
        self.cell_row = row
        self.cell_col = col
        self.tile_set = self.atlas.solid_tile_set(row, col) if self.atlas else set()
        self.neighbor_tiles = set()
        if self.atlas is not None and row + 1 < 13:
            self.neighbor_tiles = {
                (tx, ty - 45) for tx, ty in self.atlas.solid_tile_set(row + 1, col)
            }
        if clear_learned:
            self.solid_hits.clear()
            self.free_hits.clear()
            self.predicted_solid.clear()
            self.predicted_free.clear()

    def _atlas_solid(self, wx: float, wy: float) -> bool:
        if not self.tile_set and not self.neighbor_tiles:
            return False
        if not (0.0 <= wx < 480.0 and -360.0 <= wy < 360.0):
            return False
        key = (int(wx // self.cell), int(wy // self.cell))
        return key in self.tile_set or key in self.neighbor_tiles

    def is_solid(self, wx: float, wy: float) -> bool:
        if (self.tile_set or self.neighbor_tiles) and 0.0 <= wx < 480.0 and -360.0 <= wy < 360.0:
            return self._atlas_solid(wx, wy)
        key = self._cell(wx, wy)
        if key in self.solid_hits:
            return True
        if key in self.free_hits:
            return False
        return key in self.predicted_solid

    def state(self, wx: float, wy: float) -> str:
        if (self.tile_set or self.neighbor_tiles) and 0.0 <= wx < 480.0 and -360.0 <= wy < 360.0:
            return "solid" if self._atlas_solid(wx, wy) else "free"
        key = self._cell(wx, wy)
        if key in self.solid_hits:
            return "solid"
        if key in self.free_hits:
            return "free"
        if key in self.predicted_solid:
            return "solid"
        if key in self.predicted_free:
            return "free"
        return "unknown"

    def set_predictions(
        self,
        solid: set[tuple[int, int]],
        free: set[tuple[int, int]],
    ) -> None:
        self.predicted_solid = {
            k for k in solid if k not in self.free_hits and k not in self.solid_hits
        }
        self.predicted_free = {
            k for k in free if k not in self.solid_hits and k not in self.free_hits
        }

    def solid_keys(self) -> set[tuple[int, int]]:
        return set(self.solid_hits.keys())

    def free_keys(self) -> set[tuple[int, int]]:
        return set(self.free_hits.keys())

    def is_known_free(self, wx: float, wy: float) -> bool:
        return self.free_hits.get(self._cell(wx, wy), 0) > 0

    def is_unknown(self, wx: float, wy: float) -> bool:
        key = self._cell(wx, wy)
        return key not in self.solid_hits and key not in self.free_hits

    @staticmethod
    def _in_screen(key: tuple[int, int]) -> bool:
        return 0 <= key[0] < 60 and 0 <= key[1] < 45

    def mark_solid(self, wx: float, wy: float) -> None:
        key = self._cell(wx, wy)
        if not self._in_screen(key):
            return
        self.solid_hits[key] = self.solid_hits.get(key, 0) + 1
        self.free_hits.pop(key, None)

    def mark_free(self, wx: float, wy: float) -> None:
        key = self._cell(wx, wy)
        if not self._in_screen(key):
            return
        if key in self.solid_hits:
            return
        self.free_hits[key] = self.free_hits.get(key, 0) + 1

    def mark_rect_free(self, x0: float, y0: float, x1: float, y1: float) -> None:
        cx0, cy0 = self._cell(x0, y0)
        cx1, cy1 = self._cell(x1, y1)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                self.mark_free(cx * self.cell + 1, cy * self.cell + 1)

    def mark_ground(self, wx: float, wy: float, half_width: float = 6.0) -> None:
        step = self.cell / 2.0
        x = wx - half_width
        while x <= wx + half_width:
            self.mark_solid(x, wy + 2.0)
            x += step

    def known_solid_points(self) -> list[tuple[float, float]]:
        half = self.cell / 2
        return [(cx * self.cell + half, cy * self.cell + half) for cx, cy in self.solid_hits]

    def summary(self) -> dict[str, int]:
        return {
            "solid_cells": len(self.solid_hits),
            "free_cells": len(self.free_hits),
        }
