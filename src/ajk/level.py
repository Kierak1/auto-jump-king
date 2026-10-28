from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import xnb

GRID_W = 13
GRID_H = 13
SCREEN_W = 480
SCREEN_H = 360
CELL_TEXELS_W = 60
CELL_TEXELS_H = 45
TILE = 8


def _decode_texture(path: Path) -> np.ndarray:
    parsed = xnb.parse(path.read_bytes())
    _, pos = xnb.read_object_header(parsed, parsed.payload_pos)
    img, _ = xnb.read_texture(parsed, pos)
    return img


class LevelAtlas:
    def __init__(self, game_dir: Path, cache: Path | None = None) -> None:
        self.game_dir = Path(game_dir)
        self.cache = cache or (self.game_dir / "Content" / "level.xnb")
        self.alpha = self._load()
        self._cell_sets: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def _load(self) -> np.ndarray:
        img = _decode_texture(self.cache)
        return (img[:, :, 3] > 0).astype(np.uint8)

    def is_solid(self, cell_row: int, cell_col: int, local_x: float, local_y: float) -> bool:
        if not (0 <= cell_col < GRID_W and 0 <= cell_row < GRID_H):
            return False
        ix = cell_col * CELL_TEXELS_W + int(local_x // TILE)
        iy = cell_row * CELL_TEXELS_H + int(local_y // TILE)
        if not (0 <= ix < self.alpha.shape[1] and 0 <= iy < self.alpha.shape[0]):
            return False
        return bool(self.alpha[iy, ix])

    def solid_tile_set(self, cell_row: int, cell_col: int) -> set[tuple[int, int]]:
        key = (cell_row, cell_col)
        cached = self._cell_sets.get(key)
        if cached is not None:
            return cached
        if not (0 <= cell_col < GRID_W and 0 <= cell_row < GRID_H):
            return set()
        x0 = cell_col * CELL_TEXELS_W
        y0 = cell_row * CELL_TEXELS_H
        block = self.alpha[y0 : y0 + CELL_TEXELS_H, x0 : x0 + CELL_TEXELS_W]
        ys, xs = np.where(block > 0)
        tiles = {(int(x), int(y)) for x, y in zip(xs, ys, strict=False)}
        self._cell_sets[key] = tiles
        return tiles

    def cell_solid_tiles(self, cell_row: int, cell_col: int) -> set[tuple[int, int]]:
        return {(tx * TILE, ty * TILE) for tx, ty in self.solid_tile_set(cell_row, cell_col)}


class ScreenIdentifier:
    def __init__(self, game_dir: Path, cache: Path | None = None) -> None:
        self.game_dir = Path(game_dir)
        self.cache = cache or (self.game_dir / "Content" / "screens" / "midground")
        self.ids: list[int] = []
        self.thumbs: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        pre = Path("runs/assets/screens_cache.npz")
        if pre.exists():
            data = np.load(pre)
            self.ids = [int(i) for i in data["ids"]]
            self.thumbs = data["thumbs"]
            return
        ids: list[int] = []
        thumbs: list[np.ndarray] = []
        for path in sorted(self.cache.glob("*.xnb")):
            try:
                screen_id = int(path.stem)
            except ValueError:
                continue
            img = _decode_texture(path)
            if img.shape[0] != SCREEN_H or img.shape[1] not in (SCREEN_W, SCREEN_W - 1):
                continue
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (120, 90), interpolation=cv2.INTER_AREA)
            ids.append(screen_id)
            thumbs.append(gray)
        self.ids = ids
        self.thumbs = np.stack(thumbs).astype(np.float32) if thumbs else None
        pre.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pre, ids=np.array(ids), thumbs=self.thumbs)

    def identify(self, frame: np.ndarray) -> int | None:
        if self.thumbs is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (120, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = self.thumbs - gray[None, :, :]
        scores = (diff * diff).mean(axis=(1, 2))
        best = int(np.argmin(scores))
        return self.ids[best]

    def identify_cell(self, frame: np.ndarray) -> tuple[int, int] | None:
        screen_id = self.identify(frame)
        if screen_id is None:
            return None
        index = screen_id - 1
        return index % GRID_W, index // GRID_W
