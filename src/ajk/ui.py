from __future__ import annotations

import ctypes
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .controller import Controller

WINDOW = "Auto Jump King"
PANEL_BG = (28, 28, 32)
TEXT = (230, 230, 230)
ACCENT = (80, 200, 255)
GOOD = (120, 220, 120)
WARN = (80, 170, 255)
BAD = (90, 90, 240)

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


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" ") if paragraph else [""]:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


class Overlay:
    def __init__(
        self,
        frame_size: tuple[int, int] = (480, 360),
        zoom: float = 1.5,
        panel_width: int = 520,
        refresh_hz: float = 8.0,
    ) -> None:
        self.frame_w, self.frame_h = frame_size
        self.zoom = zoom
        self.panel_width = panel_width
        self.refresh_interval = 1.0 / refresh_hz
        self._last_draw = 0.0
        self._composite: np.ndarray | None = None
        self._created = False
        self.paused = False
        self.abort_requested = False

    def _on_pause(self, value: int) -> None:
        self.paused = bool(value)

    def _on_stop(self, value: int) -> None:
        if value:
            self.abort_requested = True

    def _draw_collision(self, canvas: np.ndarray, controller: Controller) -> None:
        cell = controller.collision.cell
        zoom = self.zoom
        vp = controller.view_offset if hasattr(controller, "view_offset") else (0.0, 0.0)
        cam_x, cam_y = vp
        for (cx, cy), _count in controller.collision.solid_hits.items():
            x = int((cx * cell - cam_x) * zoom)
            y = int((cy * cell - cam_y) * zoom)
            if -cell * zoom < x < canvas.shape[1] and -cell * zoom < y < canvas.shape[0]:
                cv2.rectangle(
                    canvas,
                    (x, y),
                    (x + int(cell * zoom), y + int(cell * zoom)),
                    BAD,
                    1,
                )
        for cx, cy in controller.collision.predicted_solid:
            x = int((cx * cell - cam_x) * zoom)
            y = int((cy * cell - cam_y) * zoom)
            if -cell * zoom < x < canvas.shape[1] and -cell * zoom < y < canvas.shape[0]:
                cv2.rectangle(
                    canvas,
                    (x, y),
                    (x + int(cell * zoom), y + int(cell * zoom)),
                    (0, 140, 255),
                    1,
                )

    def _draw_candidates(self, canvas: np.ndarray, controller: Controller) -> None:
        zoom = self.zoom
        cam_x, cam_y = controller.view_offset
        chosen = controller.last_choice
        for cand, plan in controller.last_candidates:
            points = [
                (int((p.x - cam_x) * zoom), int((p.y - cam_y) * zoom)) for p in plan.points[::4]
            ]
            if len(points) < 2:
                continue
            if cand.key == chosen:
                color = GOOD
                thickness = 2
            elif cand.kind == "probe":
                color = WARN
                thickness = 1
            else:
                color = (200, 200, 90)
                thickness = 1
            cv2.polylines(canvas, [np.array(points, np.int32)], False, color, thickness)
            lx, ly = points[-1]
            cv2.putText(canvas, cand.key, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_trail(self, canvas: np.ndarray, controller: Controller) -> None:
        zoom = self.zoom
        cam_x, cam_y = controller.view_offset
        points = [
            (int((p.x - cam_x) * zoom), int((p.y - cam_y) * zoom))
            for p in list(controller.history)[-80:]
        ]
        if len(points) < 2:
            return
        cv2.polylines(canvas, [np.array(points, np.int32)], False, (255, 80, 255), 1)

    def _draw_landing_markers(self, canvas: np.ndarray, controller: Controller) -> None:
        zoom = self.zoom
        cam_x, cam_y = controller.view_offset
        result = controller.last_landing_result
        if result is not None:
            planned, actual, (miss_x, miss_y) = result
            px = int((planned[0] - cam_x) * zoom)
            py = int((planned[1] - cam_y) * zoom)
            ax = int((actual[0] - cam_x) * zoom)
            ay = int((actual[1] - cam_y) * zoom)
            cv2.drawMarker(canvas, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.drawMarker(canvas, (ax, ay), GOOD, cv2.MARKER_TILTED_CROSS, 12, 1)
            cv2.line(canvas, (px, py), (ax, ay), (160, 160, 160), 1)
            cv2.putText(
                canvas,
                f"err ({miss_x:.0f},{miss_y:.0f})",
                (ax + 6, ay + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                TEXT,
                1,
            )
        plan = controller.current_plan or controller.last_plan
        if plan is not None and plan.landing is not None:
            lx = int((plan.landing[0] - cam_x) * zoom)
            ly = int((plan.landing[1] - cam_y) * zoom)
            cv2.circle(canvas, (lx, ly), 4, (0, 0, 255), 1)

    def _draw_player(self, canvas: np.ndarray, controller: Controller) -> None:
        obs = controller.prev_obs
        if obs is None:
            return
        zoom = self.zoom
        x0 = int(obs.left * zoom)
        y0 = int(obs.top * zoom)
        x1 = int(obs.right * zoom)
        y1 = int(obs.bottom * zoom)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 255, 60), 1)
        cv2.putText(
            canvas,
            f"({controller.state.x:.0f},{controller.state.y:.0f}) "
            f"v=({controller.state.vx:+.0f},{controller.state.vy:+.0f})",
            (min(x1 + 4, canvas.shape[1] - 150), y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            TEXT,
            1,
        )

    def _panel_lines(self, controller: Controller) -> list[tuple[str, tuple[int, int, int]]]:
        stats = controller.stats
        decision = controller.last_decision
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self.paused:
            lines.append(("PAUSED (slider -> 0 to resume, stop -> 1 to abort)", WARN))
        lines.append((f"backend: {controller.cfg.brain.backend}", ACCENT))
        if decision is not None:
            lines.append(
                (
                    f"choice: {decision.choice}  conf={decision.confidence:.2f}  "
                    f"latency={controller.decision_latency * 1000:.0f}ms",
                    GOOD if decision.confidence >= controller.cfg.brain.min_confidence else WARN,
                )
            )
            probs = decision.raw.get("jump", {}).get("probabilities") if decision.raw else None
            if isinstance(probs, dict):
                top = sorted(probs.items(), key=lambda kv: -float(kv[1]))[:4]
                lines.append(("probabilities:", TEXT))
                for name, value in top:
                    lines.append((f"   {name}: {float(value):.2f}", TEXT))
        lines.append(
            (
                f"jumps={stats.jumps} falls={stats.falls} "
                f"decisions={stats.decisions} best_y={stats.best_y:.0f}",
                TEXT,
            )
        )
        lines.append(
            (
                f"player=({controller.state.x:.0f},{controller.state.y:.0f}) "
                f"v=({controller.state.vx:+.0f},{controller.state.vy:+.0f}) "
                f"grounded={'yes' if controller.state.grounded else 'no'} "
                f"src={controller.state.source}",
                TEXT,
            )
        )
        if controller.memory_ok and controller.memory.address is not None:
            lines.append(
                (
                    f"memory: {controller.memory.address:#010x} "
                    f"visible={'yes' if controller.player_visible else 'no'}",
                    GOOD if controller.player_visible else WARN,
                )
            )
        lines.append(("", TEXT))
        lines.append(("state sent to model:", ACCENT))
        for line in _wrap(controller.last_state_text, 62)[:26]:
            lines.append((line, TEXT))
        return lines

    def update(self, frame: np.ndarray, controller: Controller) -> None:
        now = time.monotonic()
        if now - self._last_draw < self.refresh_interval:
            return
        self._last_draw = now
        canvas = cv2.resize(
            frame,
            (int(self.frame_w * self.zoom), int(self.frame_h * self.zoom)),
            interpolation=cv2.INTER_NEAREST,
        )
        self._draw_collision(canvas, controller)
        self._draw_trail(canvas, controller)
        self._draw_candidates(canvas, controller)
        self._draw_landing_markers(canvas, controller)
        self._draw_player(canvas, controller)
        panel = np.full((canvas.shape[0], self.panel_width, 3), PANEL_BG, np.uint8)
        y = 22
        for text, color in self._panel_lines(controller):
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 16
            if y > panel.shape[0] - 8:
                break
        self._composite = np.hstack([canvas, panel])

    def show(self) -> None:
        if self._composite is None:
            return
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

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(WINDOW)
            self._created = False

    def save(self, path: str) -> None:
        if self._composite is not None:
            cv2.imwrite(path, self._composite)

    def __enter__(self) -> Overlay:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
