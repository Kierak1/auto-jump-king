"""Small shared dataclasses used by the control loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float


@dataclass
class PlayerState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = False
    screen_x: float = 0.0
    screen_y: float = 0.0
    seen: bool = False
    source: str = "vision"


@dataclass
class ControllerStats:
    jumps: int = 0
    falls: int = 0
    decisions: int = 0
    low_confidence: int = 0
    best_y: float = 1e9
    start_time: float = field(default_factory=time.monotonic)

    def summary(self) -> dict[str, float]:
        return {
            "jumps": self.jumps,
            "falls": self.falls,
            "decisions": self.decisions,
            "low_confidence": self.low_confidence,
            "best_y": self.best_y,
            "elapsed_s": round(time.monotonic() - self.start_time, 1),
        }
