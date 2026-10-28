"""Overlay for the exact controller: game frame + world collision + plans."""

from __future__ import annotations

import ctypes
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .exact_controller import ExactController

WINDOW = "Auto Jump King (exact)"
PANEL_BG = (28, 28, 32)
TEXT = (230, 230, 230)
ACCENT = (80, 200, 255)
GOOD = (120, 220, 120)
WARN = (80, 170, 255)
ARC = (200, 200, 90)
CHOSEN_ARC = (120, 220, 120)

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080


def _make_no_activate(title: str) -> None:
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return
    get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    style = get_long(hwnd, GWL_EXSTYLE)
    set_long(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)


class OverlayExact:
    def __init__(self, frame_size=(480, 360), zoom=1.5, panel_width=560, refresh_hz=10.0):
        self.frame_w, self.frame_h = frame_size
        self.zoom = zoom
        self.panel_width = panel_width
        self.refresh_interval = 1.0 / refresh_hz
        self._last_draw = 0.0
        self._composite: np.ndarray | None = None
        self._created = False
        self._dirty = False
        self.paused = False
        self.abort_requested = False

    def _on_pause(self, value: int) -> None:
        self.paused = bool(value)

    def _on_stop(self, value: int) -> None:
        if value:
            self.abort_requested = True

    def _to_screen(self, controller: ExactController, x: float, y: float) -> tuple[int, int]:
        screen = controller.screen
        return (
            int(x * self.zoom),
            int((y + screen * 360.0) * self.zoom),
        )

    def _draw_collision(self, canvas: np.ndarray, controller: ExactController) -> None:
        screen = controller.screen
        grid = controller.grid
        zoom = self.zoom
        start_row = grid.offset - screen * 45
        if start_row < 0:
            return
        block = grid.tiles[start_row : start_row + 45]
        solid = np.isin(block, (1, 2, 3, 4, 5))
        scale = 8.0 * zoom
        if abs(scale - round(scale)) > 1e-6:
            return
        step = int(round(scale))
        mask = np.repeat(np.repeat(solid, step, axis=0), step, axis=1)
        height = min(canvas.shape[0], mask.shape[0])
        width = min(canvas.shape[1], mask.shape[1])
        region = canvas[:height, :width]
        selected = mask[:height, :width]
        tint = np.array((90, 90, 240), dtype=np.float32)
        region[selected] = (region[selected].astype(np.float32) * 0.5 + tint * 0.5).astype(np.uint8)

    def _draw_platforms(self, canvas: np.ndarray, controller: ExactController) -> None:
        screen = controller.screen
        for platform in controller.platforms:
            if platform.screen != screen:
                continue
            x0, y0 = self._to_screen(controller, platform.x0, platform.top_y)
            x1, _ = self._to_screen(controller, platform.x1, platform.top_y)
            cv2.line(canvas, (x0, y0), (x1, y0), GOOD, 1)

    def _draw_chain(self, canvas: np.ndarray, controller: ExactController) -> None:
        chain = controller.chain
        if chain is None:
            return
        for index, link in enumerate(chain.links):
            target = controller.platforms[link.target]
            if target.screen == controller.screen:
                x0, y0 = self._to_screen(controller, target.x0, target.top_y)
                x1, _ = self._to_screen(controller, target.x1, target.top_y)
                color = GOOD if link.edge.robust else WARN
                cv2.line(canvas, (x0, y0), (x1, y0), color, 2)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (x0 + 3, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            source = controller.platforms[link.source]
            if source.screen == controller.screen:
                tx, ty = self._to_screen(controller, link.edge.takeoff_x, source.top_y)
                cv2.drawMarker(canvas, (tx, ty), CHOSEN_ARC, cv2.MARKER_TRIANGLE_UP, 8, 1)
            if link.edge.landing is not None:
                lx, ly = self._to_screen(controller, link.edge.landing[0], link.edge.landing[1])
                cv2.drawMarker(canvas, (lx, ly), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)

    def _draw_arc(self, canvas: np.ndarray, controller: ExactController) -> None:
        points = controller.planned_arc
        if len(points) < 2:
            return
        screen_points = [self._to_screen(controller, x, y) for x, y in points]
        cv2.polylines(canvas, [np.array(screen_points, np.int32)], False, CHOSEN_ARC, 2)
        landing = controller.planned_landing
        if landing is not None:
            lx, ly = self._to_screen(controller, landing[0], landing[1])
            cv2.drawMarker(canvas, (lx, ly), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

    def _draw_player(self, canvas: np.ndarray, controller: ExactController) -> None:
        state = controller.state
        if state is None:
            return
        x0, y0 = self._to_screen(controller, state.x, state.y)
        x1, y1 = self._to_screen(controller, state.x + 18.0, state.y + 26.0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 255, 60), 1)

    def _draw_landing_error(self, canvas: np.ndarray, controller: ExactController) -> None:
        error = controller.last_landing_error
        if error is None:
            return
        text = f"last landing error: ({error[0]:.1f},{error[1]:.1f}) px"
        cv2.putText(canvas, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT, 1)

    def _panel_lines(self, controller: ExactController) -> list[tuple[str, tuple[int, int, int]]]:
        stats = controller.stats
        state = controller.state
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self.paused:
            lines.append(("PAUSED (slider -> 0 to resume, stop -> 1 to abort)", WARN))
        lines.append(("engine: exact physics, frame-locked input", ACCENT))
        if controller.ending_detected:
            lines.append(("ENDING REACHED!", GOOD))
        if state is not None:
            lines.append(
                (
                    f"player=({state.x:.1f},{state.y:.1f}) v=({state.vx:+.2f},{state.vy:+.2f}) "
                    f"grounded={'yes' if state.grounded else 'no'} screen={controller.screen}",
                    TEXT,
                )
            )
            address = controller.memory.address
            lines.append(
                (
                    f"memory: {'scanning' if address is None else f'{address:#010x}'} "
                    f"stamp={state.time_stamp}",
                    GOOD if address else WARN,
                )
            )
        lines.append(
            (
                f"jumps={stats.jumps} falls={stats.falls} misses={stats.misses} "
                f"decisions={stats.decisions} best_y={stats.best_y:.0f} "
                f"screen={stats.best_screen}",
                TEXT,
            )
        )
        lines.append((f"phase={controller.phase} blacklist={len(controller.blacklist)}", TEXT))
        if controller.wind_offset is not None:
            lines.append((f"wind calibrated offset={controller.wind_offset:.1f}f", ACCENT))
        elif controller.wind_enabled:
            lines.append(("wind active (uncalibrated)", WARN))
        chain = controller.chain
        if chain is not None and chain.links:
            grade = "robust" if chain.robust else "best effort"
            lines.append((f"chain: {len(chain.links)} jumps, {grade}", CHOSEN_ARC))
            for index, link in enumerate(chain.links[:12]):
                edge = link.edge
                mark = GOOD if edge.robust else TEXT
                lines.append((f"  {index + 1}. {edge.describe()}", mark))
        if controller.edge is not None:
            lines.append((f"next: {controller.edge.describe()}", CHOSEN_ARC))
        return lines

    def update(self, controller: ExactController) -> None:
        now = time.monotonic()
        if now - self._last_draw < self.refresh_interval:
            return
        self._last_draw = now
        frame = controller.current_frame
        if frame is None:
            return
        canvas = cv2.resize(
            frame,
            (int(self.frame_w * self.zoom), int(self.frame_h * self.zoom)),
            interpolation=cv2.INTER_NEAREST,
        )
        self._draw_collision(canvas, controller)
        self._draw_platforms(canvas, controller)
        self._draw_chain(canvas, controller)
        self._draw_arc(canvas, controller)
        self._draw_landing_error(canvas, controller)
        self._draw_player(canvas, controller)
        panel = np.full((canvas.shape[0], self.panel_width, 3), PANEL_BG, np.uint8)
        y = 20
        for text, color in self._panel_lines(controller):
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 15
            if y > panel.shape[0] - 6:
                break
        self._composite = np.hstack([canvas, panel])
        self._dirty = True

    def show(self) -> None:
        if self._composite is None or not self._dirty:
            return
        self._dirty = False
        if not self._created:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW, self._composite)
            cv2.waitKey(1)
            _make_no_activate(WINDOW)
            cv2.createTrackbar("pause 0/1", WINDOW, 0, 1, self._on_pause)
            cv2.createTrackbar("stop 0/1", WINDOW, 0, 1, self._on_stop)
            self._created = True
        cv2.imshow(WINDOW, self._composite)
        cv2.waitKey(1)

    def save(self, path: str) -> None:
        if self._composite is not None:
            cv2.imwrite(path, self._composite)

    def save_async(self, path: str) -> None:
        """Write the current overlay without blocking the control loop."""
        if self._composite is None:
            return
        image = self._composite.copy()
        threading.Thread(target=cv2.imwrite, args=(path, image), daemon=True).start()

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(WINDOW)
            self._created = False

    def __enter__(self) -> OverlayExact:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
