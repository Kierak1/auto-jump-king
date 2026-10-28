from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import mss
import numpy as np

from .win import WindowInfo, refresh


@dataclass
class RecordedFrame:
    t: float
    frame: np.ndarray


@dataclass
class FrameRecorder:
    window: WindowInfo
    target_width: int = 480
    max_frames: int = 900
    _frames: list[RecordedFrame] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _t0: float = 0.0

    def start(self) -> None:
        self._frames = []
        self._stop.clear()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        with mss.mss() as sct:
            while not self._stop.is_set() and len(self._frames) < self.max_frames:
                info = refresh(self.window)
                monitor = {
                    "left": info.client_x,
                    "top": info.client_y,
                    "width": info.client_w,
                    "height": info.client_h,
                }
                raw = np.asarray(sct.grab(monitor), dtype=np.uint8)
                step = max(1, round(info.client_w / self.target_width))
                frame = np.ascontiguousarray(raw[::step, ::step, :3])
                self._frames.append(RecordedFrame(time.monotonic() - self._t0, frame))

    def stop(self) -> list[RecordedFrame]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self._frames)

    @property
    def count(self) -> int:
        return len(self._frames)

    @property
    def fps(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        dt = self._frames[-1].t - self._frames[0].t
        return (len(self._frames) - 1) / dt if dt > 0 else 0.0
