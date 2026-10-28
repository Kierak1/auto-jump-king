from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

CELL = 8


def cell_features(frame: np.ndarray, cell: int = CELL) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    gh, gw = h // cell, w // cell
    crop = frame[: gh * cell, : gw * cell].astype(np.float32)
    cells = (
        crop.reshape(gh, cell, gw, cell, 3).transpose(0, 2, 1, 3, 4).reshape(gh, gw, cell * cell, 3)
    )
    mean = cells.mean(axis=2)
    std = cells.std(axis=2)
    return np.concatenate([mean, std], axis=2), gh, gw


@dataclass
class VisualSolidClassifier:
    cell: int = CELL
    k: int = 5
    max_examples: int = 600
    examples: deque[tuple[np.ndarray, bool]] = field(default_factory=lambda: deque(maxlen=600))
    _keys: set[tuple[int, int, bool]] = field(default_factory=set)

    def _world_cell(
        self, frame_x: int, frame_y: int, cam_x: float, cam_y: float
    ) -> tuple[int, int]:
        wx = frame_x + cam_x
        wy = frame_y + cam_y
        return int(np.floor(wx / self.cell)), int(np.floor(wy / self.cell))

    def observe(
        self,
        frame: np.ndarray,
        cam_x: float,
        cam_y: float,
        solid_cells: set[tuple[int, int]],
        free_cells: set[tuple[int, int]],
        occluded: tuple[float, float, float, float] | None = None,
    ) -> int:
        features, gh, gw = cell_features(frame, self.cell)
        added = 0
        items: list[tuple[tuple[int, int], bool]] = [(k, True) for k in solid_cells]
        items += [(k, False) for k in free_cells]
        for key, label in items:
            if (key[0], key[1], label) in self._keys:
                continue
            fx = key[0] * self.cell + self.cell / 2 - cam_x
            fy = key[1] * self.cell + self.cell / 2 - cam_y
            if occluded is not None:
                x0, y0, x1, y1 = occluded
                inside_x = x0 - self.cell <= fx <= x1 + self.cell
                inside_y = y0 - self.cell <= fy <= y1 + self.cell
                if inside_x and inside_y:
                    continue
            gx, gy = int(fx // self.cell), int(fy // self.cell)
            if not (0 <= gx < gw and 0 <= gy < gh):
                continue
            self._keys.add((key[0], key[1], label))
            self.examples.append((features[gy, gx].copy(), label))
            added += 1
        return added

    def predict(
        self,
        frame: np.ndarray,
        cam_x: float,
        cam_y: float,
        solid_threshold: float = 0.7,
        free_threshold: float = 0.3,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        if len(self.examples) < 12:
            return set(), set()
        features, gh, gw = cell_features(frame, self.cell)
        flat = features.reshape(-1, features.shape[2])
        train_x = np.stack([e[0] for e in self.examples])
        train_y = np.array([1.0 if e[1] else 0.0 for e in self.examples])
        scale = train_x.std(axis=0) + 1e-3
        diff = (flat[:, None, :] - train_x[None, :, :]) / scale[None, None, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        k = min(self.k, len(train_y))
        idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        votes = train_y[idx].mean(axis=1)
        predicted_solid: set[tuple[int, int]] = set()
        predicted_free: set[tuple[int, int]] = set()
        for gy in range(gh):
            for gx in range(gw):
                p = votes[gy * gw + gx]
                key = self._world_cell(gx * self.cell, gy * self.cell, cam_x, cam_y)
                if p >= solid_threshold:
                    predicted_solid.add(key)
                elif p <= free_threshold:
                    predicted_free.add(key)
        return predicted_solid, predicted_free
