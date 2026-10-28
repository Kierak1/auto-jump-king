from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .assets import KING_BODY_PALETTE

BLUE_PALETTE_INDICES = [2, 4]


@dataclass
class PlayerObs:
    cx: float
    bottom: float
    left: float
    top: float
    right: float
    score: float
    source: str = "palette"
    area: int = 0
    merged: bool = False


@dataclass
class _Candidate:
    x: int
    y: int
    w: int
    h: int
    area: int
    blue: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def bottom(self) -> float:
        return float(self.y + self.h)


class PlayerDetector:
    def __init__(
        self,
        palette: np.ndarray = KING_BODY_PALETTE,
        blue_indices: list[int] | None = None,
        tolerance: int = 10,
        min_area: int = 60,
        min_blue_fraction: float = 0.15,
        min_blue_pixels: int = 20,
    ) -> None:
        self.palette = palette.astype(np.int16)
        idx = BLUE_PALETTE_INDICES if blue_indices is None else blue_indices
        self.blue_palette = self.palette[idx]
        self.tolerance = tolerance
        self.min_area = min_area
        self.min_blue_fraction = min_blue_fraction
        self.min_blue_pixels = min_blue_pixels
        self._bounds = [
            (
                np.clip(c - tolerance, 0, 255).astype(np.uint8),
                np.clip(c + tolerance, 0, 255).astype(np.uint8),
            )
            for c in self.palette
        ]
        self._blue_bounds = [
            (
                np.clip(c - tolerance, 0, 255).astype(np.uint8),
                np.clip(c + tolerance, 0, 255).astype(np.uint8),
            )
            for c in self.blue_palette
        ]

    def _masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask_all = None
        for low, high in self._bounds:
            m = cv2.inRange(frame, low, high)
            mask_all = m if mask_all is None else cv2.bitwise_or(mask_all, m)
        mask_blue = None
        for low, high in self._blue_bounds:
            m = cv2.inRange(frame, low, high)
            mask_blue = m if mask_blue is None else cv2.bitwise_or(mask_blue, m)
        assert mask_all is not None and mask_blue is not None
        return mask_all, mask_blue

    def _components(self, frame: np.ndarray) -> list[_Candidate]:
        mask_all, mask_blue = self._masks(frame)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_all, connectivity=8)
        candidates: list[_Candidate] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.min_area:
                continue
            component = labels[y : y + h, x : x + w] == i
            blue_mask = mask_blue[y : y + h, x : x + w] & component
            blue = int(np.count_nonzero(blue_mask))
            if blue < self.min_blue_pixels:
                continue
            ys, xs = np.where(blue_mask > 0)
            bx0 = x + int(xs.min())
            by0 = y + int(ys.min())
            bx1 = x + int(xs.max())
            by1 = y + int(ys.max())
            candidates.append(_Candidate(bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1, area, blue))
        return candidates

    def detect(
        self,
        frame: np.ndarray,
        prev: PlayerObs | None = None,
        max_distance: float = 80.0,
    ) -> PlayerObs | None:
        candidates = self._components(frame)
        if not candidates:
            return None
        height, width = frame.shape[:2]

        def clipped(c: _Candidate) -> bool:
            return c.x <= 1 or c.y <= 1 or c.x + c.w >= width - 1 or c.y + c.h >= height - 1

        plausible = [
            c
            for c in candidates
            if c.blue >= self.min_blue_pixels
            and c.blue / max(1, c.w * c.h) >= self.min_blue_fraction
            and (
                (12 <= c.w <= 48 and 16 <= c.h <= 64)
                or (4 <= c.w <= 48 and 6 <= c.h <= 64)
                or (clipped(c) and c.w >= 6 and c.h >= 12)
            )
        ]
        if not plausible:
            return None
        if prev is not None:
            close = [
                c
                for c in plausible
                if abs(c.cx - prev.cx) <= max_distance
                and abs(c.bottom - prev.bottom) <= max_distance
            ]
            if close:
                plausible = close
        best = max(plausible, key=lambda c: c.blue)
        total = sum(c.blue for c in plausible) or 1
        return PlayerObs(
            cx=best.cx,
            bottom=best.bottom,
            left=float(best.x),
            top=float(best.y),
            right=float(best.x + best.w),
            score=best.blue / total,
            area=best.area,
            merged=best.area > 600,
        )


class GroundSensor:
    def __init__(self, rim_threshold: float = 200.0, below_max: float = 170.0) -> None:
        self.rim_threshold = rim_threshold
        self.below_max = below_max

    def find_surface_y(
        self,
        frame: np.ndarray,
        frame_x: int,
        center_y: int,
        search: int = 5,
    ) -> int | None:
        if not (0 <= frame_x < frame.shape[1]):
            return None
        y0 = max(0, center_y - search)
        y1 = min(frame.shape[0], center_y + search + 1)
        if y1 <= y0:
            return None
        column = frame[y0:y1, frame_x].astype(np.float32).sum(axis=1)
        idx = int(np.argmax(column))
        y = y0 + idx
        if column[idx] < self.rim_threshold:
            return None
        below = frame[y + 3 : y + 10, frame_x].astype(np.float32).sum(axis=1)
        if below.size and float(below.mean()) > self.below_max:
            return None
        return y

    def ground_cells(
        self,
        frame: np.ndarray,
        cam_x: float,
        cam_y: float,
        world_x: float,
        world_y: float,
        span: float = 96.0,
        step: float = 8.0,
        skip_half: float = 14.0,
    ) -> list[tuple[float, float]]:
        cells: list[tuple[float, float]] = []
        dx = -span
        while dx <= span:
            if abs(dx) >= skip_half:
                frame_x = int(round(world_x + dx - cam_x))
                frame_y = int(round(world_y - cam_y))
                surface = self.find_surface_y(frame, frame_x, frame_y)
                if surface is not None:
                    world_surface = surface + cam_y
                    if abs(world_surface - world_y) <= 4:
                        cells.append((world_x + dx, world_surface + 3.0))
            dx += step
        return cells


class SurfaceScanner:
    def __init__(
        self,
        brightness: float = 200.0,
        below_max: float = 180.0,
        min_run: int = 24,
        step: int = 4,
    ) -> None:
        self.brightness = brightness
        self.below_max = below_max
        self.min_run = min_run
        self.step = step

    def scan(
        self,
        frame: np.ndarray,
        occluded: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int]]:
        pixels = frame.astype(np.float32).sum(axis=2)
        bright = pixels > self.brightness
        h = pixels.shape[0]
        below_sum = np.zeros_like(pixels)
        for offset in range(3, 11):
            below_sum[: h - offset] += pixels[offset:]
        below_mean = below_sum / 8.0
        contrast = bright & (pixels > below_mean * 1.7)
        below_ok = (below_mean < self.below_max) & (
            np.roll(below_mean, -4, axis=0) < self.below_max
        )
        candidate = contrast & below_ok
        if occluded is not None:
            x0, y0, x1, y1 = occluded
            candidate[max(0, y0 - 2) : y1 + 2, max(0, x0 - 2) : x1 + 2] = False
        padded = np.pad(candidate.astype(np.int8), ((0, 0), (1, 1)))
        diff = np.diff(padded, axis=1)
        starts = np.argwhere(diff == 1)
        ends = np.argwhere(diff == -1)
        surfaces: list[tuple[int, int]] = []
        for (row, x_start), (row2, x_end) in zip(starts, ends, strict=False):
            if row != row2:
                continue
            length = x_end - x_start
            if length < self.min_run:
                continue
            for x in range(int(x_start), int(x_end), self.step):
                surfaces.append((x, int(row) + 4))
        return surfaces


@dataclass
class CameraUpdate:
    dx: float
    dy: float
    ok: bool


class CameraTracker:
    def __init__(self, frame_shape: tuple[int, int]) -> None:
        self.h, self.w = frame_shape
        self.prev_gray: np.ndarray | None = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._frac_x = 0.0
        self._frac_y = 0.0
        self.last = CameraUpdate(0.0, 0.0, False)

    @staticmethod
    def _prepare(gray: np.ndarray, mask_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        prepared = gray.copy()
        for x0, y0, x1, y1 in mask_boxes:
            x0 = max(0, x0)
            y0 = max(0, y0)
            prepared[y0:y1, x0:x1] = 0
        return prepared

    def realign(self, x: float, y: float) -> None:
        self._frac_x = x
        self._frac_y = y
        self.offset_x = round(x)
        self.offset_y = round(y)

    def update(
        self,
        frame: np.ndarray,
        mask_boxes: list[tuple[int, int, int, int]] | None = None,
        min_response: float = 0.05,
    ) -> CameraUpdate:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        boxes = mask_boxes or []
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = self._prepare(gray, boxes)
            self.last = CameraUpdate(0.0, 0.0, False)
            return self.last
        current = self._prepare(gray, boxes)
        window = cv2.createHanningWindow((self.w, self.h), cv2.CV_32F)
        try:
            (dx, dy), response = cv2.phaseCorrelate(self.prev_gray, current, window)
        except cv2.error:
            self.prev_gray = current
            self.last = CameraUpdate(0.0, 0.0, False)
            return self.last
        ok = response >= min_response and abs(dx) < self.w / 3 and abs(dy) < self.h / 3
        if ok:
            self._frac_x += dx
            self._frac_y += dy
            self.offset_x = round(self._frac_x)
            self.offset_y = round(self._frac_y)
        self.prev_gray = current
        self.last = CameraUpdate(float(dx), float(dy), bool(ok))
        return self.last
