from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RunLogger:
    def __init__(self, runs_dir: str | Path, name: str = "run") -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(runs_dir) / f"{name}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self._handle = self.events_path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()

    def event(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.monotonic() - self._t0, 3), "kind": kind}
        record.update(data)
        self._handle.write(json.dumps(record, default=float) + "\n")
        self._handle.flush()

    def frame(self, frame: np.ndarray, label: str) -> None:
        # PNG encoding costs tens of ms; write it off the control loop.
        path = self.dir / f"{label}.png"
        image = frame.copy()
        threading.Thread(target=cv2.imwrite, args=(str(path), image), daemon=True).start()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
