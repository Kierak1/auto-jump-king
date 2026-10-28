from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actions import Keyboard
from .capture import GameView
from .record import FrameRecorder
from .vision import CameraTracker, PlayerDetector, PlayerObs
from .win import WindowInfo, focus


@dataclass
class JumpSample:
    t: float
    screen_x: float
    screen_y: float
    world_x: float
    world_y: float


@dataclass
class JumpTrace:
    charge_s: float
    x0: float
    y0: float
    samples: list[JumpSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "charge_s": self.charge_s,
            "x0": self.x0,
            "y0": self.y0,
            "samples": [s.__dict__ for s in self.samples],
        }


@dataclass
class DriftTrace:
    charge_s: float
    direction: int
    x0: float
    y0: float
    samples: list[JumpSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "charge_s": self.charge_s,
            "direction": self.direction,
            "x0": self.x0,
            "y0": self.y0,
            "samples": [s.__dict__ for s in self.samples],
        }


def _walk_to_x(
    view: GameView,
    detector: PlayerDetector,
    kb: Keyboard,
    target_x: float,
    timeout_s: float = 4.0,
) -> PlayerObs | None:
    obs = detector.detect(view.grab())
    if obs is None:
        return None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        error = target_x - obs.cx
        if abs(error) < 6:
            break
        key = "RIGHT" if error > 0 else "LEFT"
        kb.down(key)
        time.sleep(0.05)
        kb.up(key)
        time.sleep(0.02)
        new_obs = detector.detect(view.grab(), obs)
        if new_obs is not None:
            obs = new_obs
    kb.release_all()
    time.sleep(0.25)
    return detector.detect(view.grab(), obs)


def _measure_drift_jump(
    view: GameView,
    detector: PlayerDetector,
    tracker: CameraTracker,
    kb: Keyboard,
    charge_s: float,
    direction_key: str,
    start_x: float = 200.0,
    sample_s: float = 1.6,
    settle_s: float = 0.4,
) -> DriftTrace | None:
    kb.release_all()
    time.sleep(settle_s)
    obs = _walk_to_x(view, detector, kb, start_x)
    if obs is None:
        return None
    time.sleep(0.3)
    obs = detector.detect(view.grab(), obs) or obs
    trace = DriftTrace(charge_s=charge_s, direction=1, x0=obs.cx, y0=obs.bottom)
    recorder = FrameRecorder(view.window)
    recorder.start()
    time.sleep(0.15)
    kb.down(direction_key)
    time.sleep(0.06)
    kb.down("SPACE")
    time.sleep(charge_s)
    kb.up("SPACE")
    time.sleep(max(0.0, sample_s - charge_s - 0.2))
    kb.release_all()
    frames = recorder.stop()
    if not frames:
        return None
    t0 = frames[0].t
    prev = obs
    for recorded in frames:
        obs = detector.detect(recorded.frame, prev)
        if obs is None:
            continue
        prev = obs
        mask = [
            (
                int(obs.left) - 2,
                int(obs.top) - 2,
                int(obs.right) + 2,
                int(obs.bottom) + 2,
            )
        ]
        tracker.update(recorded.frame, mask)
        t_rel = recorded.t - t0
        if t_rel < 0:
            continue
        trace.samples.append(
            JumpSample(
                t=t_rel,
                screen_x=obs.cx,
                screen_y=obs.bottom,
                world_x=obs.cx + tracker.offset_x,
                world_y=obs.bottom + tracker.offset_y,
            )
        )
    return trace


def fit_drift(traces: list[DriftTrace]) -> dict[str, float]:
    gravities: list[float] = []
    air_speeds: list[float] = []
    for trace in traces:
        ys = np.array([s.world_y for s in trace.samples])
        xs = np.array([s.world_x for s in trace.samples])
        ts = np.array([s.t for s in trace.samples])
        if len(ys) < 10:
            continue
        baseline = float(np.median(ys[:4]))
        start = None
        for i in range(len(ys) - 4):
            if abs(ys[i] - baseline) < 2.0 and all(
                ys[i + k] < ys[i + k - 1] - 0.5 for k in (1, 2, 3)
            ):
                start = i
                break
        if start is None:
            continue
        t_start = ts[start]
        window = (ts >= t_start + 0.08) & (ts <= t_start + 0.40)
        if int(window.sum()) < 6:
            continue
        t = ts[window] - t_start
        y = ys[window]
        x = xs[window]
        cy = np.polyfit(t, y, 2)
        gravity = 2.0 * cy[0]
        cx = np.polyfit(t, x, 1)
        speed = abs(cx[0])
        residual = float(np.abs(np.polyval(cx, t) - x).mean())
        if 300.0 < gravity < 2000.0 and 30.0 < speed < 400.0 and residual < 4.0:
            gravities.append(float(gravity))
            air_speeds.append(float(speed))
    return {
        "gravity": float(np.median(gravities)) if gravities else 0.0,
        "air_speed": float(np.median(air_speeds)) if air_speeds else 0.0,
        "samples": float(len(air_speeds)),
    }


def run_drift_calibration(
    window: WindowInfo,
    charges: list[float],
    output: Path | None = None,
) -> tuple[list[DriftTrace], dict[str, float]]:
    detector = PlayerDetector()
    kb = Keyboard()
    focus(window.hwnd)
    traces: list[DriftTrace] = []
    with GameView(window) as view:
        view.calibrate_viewport()
        first = view.grab()
        tracker = CameraTracker(first.shape[:2])
        for charge in charges:
            trace = _measure_drift_jump(view, detector, tracker, kb, charge, "RIGHT")
            if trace is None:
                print(f"charge={charge:.2f}s: player not found, skipped")
                continue
            if trace.samples:
                ys = [s.world_y for s in trace.samples]
                xs = [s.world_x for s in trace.samples]
                print(
                    f"charge={charge:.2f}s samples={len(trace.samples)} "
                    f"dx={xs[-1] - xs[0]:+.0f}px dy={trace.y0 - min(ys):.0f}px"
                )
            traces.append(trace)
            kb.release_all()
            time.sleep(0.3)
    fit = fit_drift(traces)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {"fit": fit, "traces": [t.to_dict() for t in traces]},
                indent=2,
            )
        )
    return traces, fit


def _measure_jump(
    view: GameView,
    detector: PlayerDetector,
    tracker: CameraTracker,
    kb: Keyboard,
    charge_s: float,
    sample_s: float = 2.0,
    settle_s: float = 0.6,
    jump_key: str = "SPACE",
) -> JumpTrace | None:
    kb.release_all()
    time.sleep(settle_s)
    obs = detector.detect(view.grab())
    if obs is None:
        return None
    trace = JumpTrace(charge_s=charge_s, x0=obs.cx, y0=obs.bottom)
    recorder = FrameRecorder(view.window)
    recorder.start()
    time.sleep(0.15)
    kb.down(jump_key)
    t_press = time.monotonic()
    time.sleep(charge_s)
    kb.up(jump_key)
    time.sleep(max(0.0, sample_s - charge_s - 0.15))
    kb.release_all()
    frames = recorder.stop()
    if not frames:
        return None
    t0 = frames[0].t
    prev = obs
    for recorded in frames:
        obs = detector.detect(recorded.frame, prev)
        if obs is None:
            continue
        prev = obs
        mask = [
            (
                int(obs.left) - 2,
                int(obs.top) - 2,
                int(obs.right) + 2,
                int(obs.bottom) + 2,
            )
        ]
        tracker.update(recorded.frame, mask)
        t_rel = recorded.t - t0
        if t_rel < 0:
            continue
        trace.samples.append(
            JumpSample(
                t=t_rel,
                screen_x=obs.cx,
                screen_y=obs.bottom,
                world_x=obs.cx + tracker.offset_x,
                world_y=obs.bottom + tracker.offset_y,
            )
        )
    _ = t_press
    return trace


def _measure_run_speed(
    view: GameView,
    detector: PlayerDetector,
    kb: Keyboard,
    direction_keys: tuple[str, ...] = ("RIGHT", "LEFT"),
    run_s: float = 1.0,
) -> float:
    best = 0.0
    for key in direction_keys:
        kb.release_all()
        time.sleep(0.5)
        obs = detector.detect(view.grab())
        if obs is None:
            continue
        prev = obs
        samples: list[tuple[float, float]] = []
        kb.down(key)
        t0 = time.monotonic()
        while time.monotonic() - t0 < run_s:
            obs = detector.detect(view.grab(), prev)
            if obs is not None:
                prev = obs
                samples.append((time.monotonic() - t0, obs.cx))
        kb.release_all()
        if len(samples) < 5:
            continue
        window = [s for s in samples if s[0] > run_s * 0.4]
        if len(window) < 2:
            continue
        dt = window[-1][0] - window[0][0]
        if dt <= 0:
            continue
        speed = abs((window[-1][1] - window[0][1]) / dt)
        best = max(best, speed)
    kb.release_all()
    return float(best)


def _airborne_segment(
    trace: JumpTrace, baseline: float | None = None, min_rise: float = 4.0
) -> tuple[int, int]:
    ys = [s.world_y for s in trace.samples]
    if len(ys) < 6:
        return 0, 0
    if baseline is None:
        baseline = float(np.median(ys[: max(3, len(ys) // 10)]))
    takeoff = 0
    for i, y in enumerate(ys):
        if baseline - y > min_rise:
            takeoff = i
            break
    if takeoff == 0:
        return 0, 0
    apex = int(np.argmin(ys))
    end = apex
    for i in range(apex, len(ys)):
        if ys[i] > baseline - min_rise:
            end = i
            break
        end = i
    return takeoff, end


def fit_physics(traces: list[JumpTrace]) -> tuple[float, dict[float, float]]:
    v0s: dict[float, float] = {}
    gravities: list[float] = []
    for trace in traces:
        takeoff, end = _airborne_segment(trace)
        if end - takeoff < 6:
            continue
        segment = trace.samples[takeoff : end + 1]
        ts = np.array([s.t for s in segment])
        ys = np.array([s.world_y for s in segment])
        ts = ts - ts[0]
        coeffs = np.polyfit(ts, ys, 2)
        accel = float(2.0 * coeffs[0])
        v0 = float(-coeffs[1])
        if accel <= 0:
            continue
        gravities.append(-accel)
        v0s[trace.charge_s] = abs(v0)
    gravity = float(np.median(gravities)) if gravities else 0.0
    return gravity, v0s


@dataclass
class CalibrationResult:
    traces: list[JumpTrace]
    gravity: float
    jump_v0: dict[float, float]
    max_run_speed: float

    def to_dict(self) -> dict:
        return {
            "gravity": self.gravity,
            "jump_v0": {str(k): v for k, v in self.jump_v0.items()},
            "max_run_speed": self.max_run_speed,
            "traces": [t.to_dict() for t in self.traces],
        }


def run_calibration(
    window: WindowInfo,
    charges: list[float],
    sample_s: float = 2.0,
    output: Path | None = None,
    measure_run: bool = True,
) -> CalibrationResult:
    detector = PlayerDetector()
    kb = Keyboard()
    focus(window.hwnd)
    traces: list[JumpTrace] = []
    with GameView(window) as view:
        view.calibrate_viewport()
        first = view.grab()
        tracker = CameraTracker(first.shape[:2])
        for charge in charges:
            trace = _measure_jump(view, detector, tracker, kb, charge, sample_s=sample_s)
            if trace is None:
                print(f"charge={charge:.2f}s: player not found, skipped")
                continue
            apex = min((s.world_y for s in trace.samples), default=trace.y0)
            print(
                f"charge={charge:.2f}s samples={len(trace.samples)} height={trace.y0 - apex:.1f}px"
            )
            traces.append(trace)
        run_speed = _measure_run_speed(view, detector, kb) if measure_run else 0.0
    gravity, v0s = fit_physics(traces)
    result = CalibrationResult(traces, gravity, v0s, run_speed)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2))
    return result
