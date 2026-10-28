from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np

from .win import WindowInfo, focus, is_foreground, refresh


@dataclass(frozen=True)
class Viewport:
    x: int
    y: int
    w: int
    h: int


def detect_viewport(frame: np.ndarray, threshold: int = 8) -> Viewport:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return Viewport(0, 0, frame.shape[1], frame.shape[0])
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return Viewport(x0, y0, x1 - x0 + 1, y1 - y0 + 1)


MENU_DARK_FRACTION = 0.6
MENU_PIXEL_MAX = 24


def is_menu_frame(frame: np.ndarray) -> bool:
    """The main menu is a black screen with a light logo; the level's dark
    shaft screens stay below ~0.4 near-black pixels, the menu is at ~0.76+."""
    if frame is None or frame.size == 0:
        return False
    flat = frame.reshape(-1, frame.shape[2]) if frame.ndim == 3 else frame.reshape(-1, 1)
    dark = (flat.max(axis=1) < MENU_PIXEL_MAX).mean()
    return bool(dark >= MENU_DARK_FRACTION)


class GameView:
    def __init__(
        self,
        window: WindowInfo,
        scale: float = 1.0,
        viewport: Viewport | None = None,
        focus_timeout_s: float = 2.0,
        native_size: tuple[int, int] = (480, 360),
    ) -> None:
        self.window = window
        self.scale = scale
        self.focus_timeout_s = focus_timeout_s
        self.viewport = viewport
        self.native_size = native_size
        self._sct = mss.mss()

    def focus(self) -> bool:
        return focus(self.window.hwnd)

    def is_focused(self) -> bool:
        return is_foreground(self.window.hwnd)

    def _grab_raw(self) -> np.ndarray:
        info = refresh(self.window)
        if info.client_w <= 0 or info.client_h <= 0:
            raise RuntimeError("game window has no client area")
        monitor = {
            "left": info.client_x,
            "top": info.client_y,
            "width": info.client_w,
            "height": info.client_h,
        }
        for attempt in range(3):
            try:
                raw = np.asarray(self._sct.grab(monitor), dtype=np.uint8)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.1)
        return np.ascontiguousarray(raw[:, :, :3])

    def calibrate_viewport(self) -> Viewport:
        deadline = time.monotonic() + self.focus_timeout_s
        while time.monotonic() < deadline:
            if self.focus():
                break
            time.sleep(0.05)
        info = refresh(self.window)
        target = self.native_size[0] / self.native_size[1]
        aspect = info.client_w / max(1, info.client_h)
        if abs(aspect - target) / target < 0.03:
            self.viewport = Viewport(0, 0, info.client_w, info.client_h)
        else:
            if aspect > target:
                width = int(round(info.client_h * target))
                self.viewport = Viewport((info.client_w - width) // 2, 0, width, info.client_h)
            else:
                height = int(round(info.client_w / target))
                self.viewport = Viewport(0, (info.client_h - height) // 2, info.client_w, height)
        return self.viewport

    def grab(self, ensure_focus: bool = True) -> np.ndarray:
        if ensure_focus and not self.is_focused():
            self.focus()
        raw = self._grab_raw()
        if self.viewport is None:
            self.viewport = detect_viewport(raw)
        vp = self.viewport
        frame = raw[vp.y : vp.y + vp.h, vp.x : vp.x + vp.w]
        target_w = int(round(self.native_size[0] * self.scale))
        target_h = int(round(self.native_size[1] * self.scale))
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            interpolation = cv2.INTER_AREA if frame.shape[1] >= target_w else cv2.INTER_NEAREST
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> GameView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
