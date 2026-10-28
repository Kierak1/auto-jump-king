from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from .actions import Keyboard
from .calibrate import run_calibration, run_drift_calibration
from .capture import GameView
from .config import load_config
from .controller import Controller
from .telemetry import RunLogger
from .ui import Overlay
from .vision import CameraTracker, PlayerDetector
from .win import find_window, focus, refresh, wait_for_window


def cmd_info(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = find_window(cfg.window.process_name, cfg.window.title)
    if info is None:
        print("game window not found")
        return 1
    print(
        f"hwnd={info.hwnd} pid={info.pid} title={info.title!r} "
        f"client=({info.client_x},{info.client_y},{info.client_w}x{info.client_h})"
    )
    return 0


def cmd_focus(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    with GameView(info) as view:
        ok = view.focus()
        print("focused" if ok else "could not focus")
    return 0 if ok else 1


def cmd_shot(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    out = Path(args.out) if args.out else Path(cfg.play.runs_dir) / "shots" / "shot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    with GameView(info, scale=args.scale) as view:
        view.calibrate_viewport()
        frame = view.grab()
        vp = view.viewport
        assert vp is not None
        cv2.imwrite(str(out), frame)
        print(f"viewport={vp} scale={args.scale} saved={out} shape={frame.shape}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    out_dir = Path(cfg.play.runs_dir) / "shots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with GameView(info, scale=args.scale) as view:
        view.calibrate_viewport()
        deadline = time.monotonic() + args.duration
        n = 0
        while time.monotonic() < deadline:
            frame = view.grab()
            path = out_dir / f"frame_{int(time.time() * 1000)}.png"
            cv2.imwrite(str(path), frame)
            print(path)
            n += 1
            time.sleep(args.interval)
        print(f"captured {n} frame(s)")
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    detector = PlayerDetector()
    out_dir = Path(cfg.play.runs_dir) / "track"
    out_dir.mkdir(parents=True, exist_ok=True)
    with GameView(info, scale=args.scale) as view:
        view.calibrate_viewport()
        tracker: CameraTracker | None = None
        prev = None
        deadline = time.monotonic() + args.duration
        n = 0
        while time.monotonic() < deadline:
            frame = view.grab()
            if tracker is None:
                tracker = CameraTracker(frame.shape[:2])
            obs = detector.detect(frame, prev)
            boxes = []
            if obs is not None:
                boxes = [
                    (
                        int(obs.left) - 2,
                        int(obs.top) - 2,
                        int(obs.right) + 2,
                        int(obs.bottom) + 2,
                    )
                ]
            cam = tracker.update(frame, boxes)
            if obs is not None:
                prev = obs
                wx = obs.cx + tracker.offset_x
                wy = obs.bottom + tracker.offset_y
                print(
                    f"frame={n} score={obs.score:.4f} "
                    f"screen=({obs.cx:.1f},{obs.bottom:.1f}) "
                    f"cam=({tracker.offset_x:.1f},{tracker.offset_y:.1f}) "
                    f"world=({wx:.1f},{wy:.1f}) "
                    f"dcam=({cam.dx:.2f},{cam.dy:.2f}) ok={cam.ok}"
                )
            else:
                print(f"frame={n} player not found dcam=({cam.dx:.2f},{cam.dy:.2f})")
            if args.save_frames:
                vis = frame.copy()
                if obs is not None:
                    cv2.rectangle(
                        vis,
                        (int(obs.left), int(obs.top)),
                        (int(obs.right), int(obs.bottom)),
                        (0, 0, 255),
                        1,
                    )
                cv2.imwrite(str(out_dir / f"track_{n:04d}.png"), vis)
            n += 1
            time.sleep(args.interval)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    charges = [float(c) for c in args.charges.split(",")]
    out = Path(args.out) if args.out else Path(cfg.play.runs_dir) / "calibration" / "physics.json"
    result = run_calibration(info, charges, sample_s=args.sample_s, output=out)
    print(f"gravity={result.gravity:.1f} px/s^2")
    for charge, v0 in sorted(result.jump_v0.items()):
        print(f"  charge={charge:.2f}s v0={v0:.1f} px/s")
    print(f"max_run_speed={result.max_run_speed:.1f} px/s")
    print(f"saved to {out}")
    return 0


def _grab_frame(info) -> np.ndarray | None:
    try:
        fresh = refresh(info)
        with mss.mss() as sct:
            raw = np.asarray(
                sct.grab(
                    {
                        "left": fresh.client_x,
                        "top": fresh.client_y,
                        "width": fresh.client_w,
                        "height": fresh.client_h,
                    }
                ),
                dtype=np.uint8,
            )[:, :, :3]
        return cv2.resize(raw, (480, 360))
    except Exception:
        return None


def _run_play(
    cfg, info, *, brain: str | None, minutes: float | None, quiet: bool, no_ui: bool, legacy: bool
) -> int:
    if brain:
        cfg.brain.backend = brain
    if minutes is not None:
        cfg.play.max_minutes = minutes
    logger = RunLogger(cfg.play.runs_dir, "play")
    print(f"run dir: {logger.dir}")
    if legacy:
        controller = Controller(cfg, info, logger=logger)
        controller.verbose = not quiet
        ui = Overlay() if not no_ui else None
    else:
        from .exact_controller import ExactController
        from .ui_exact import OverlayExact

        controller = ExactController(cfg, info, logger=logger)
        controller.verbose = not quiet
        ui = OverlayExact() if not no_ui else None
    try:
        stats = controller.run(ui=ui)
    finally:
        logger.close()
    print(stats.summary())
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    return _run_play(
        cfg,
        info,
        brain=args.brain,
        minutes=args.minutes,
        quiet=args.quiet,
        no_ui=args.no_ui,
        legacy=args.legacy,
    )


def cmd_autostart(args: argparse.Namespace) -> int:
    """One-click entry: launch the game if needed, pass the title screen, play."""
    cfg = load_config(args.config)
    info = find_window(cfg.window.process_name, cfg.window.title)
    if info is None:
        exe = Path(cfg.game_dir) / "JumpKing.exe"
        if not exe.exists():
            print(f"game executable not found: {exe}")
            return 1
        print(f"launching {exe}")
        subprocess.Popen([str(exe)], cwd=cfg.game_dir)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=45.0)
    if info is None:
        print("game window not found")
        return 1
    ctypes.windll.user32.ShowWindow(info.hwnd, 9)  # SW_RESTORE
    focus(info.hwnd)
    time.sleep(1.0)
    keyboard = Keyboard(tap_ms=80)
    # Fast menu skip: the main menu is a black screen, so poll the frame at
    # ~4 Hz and tap SPACE until it clears instead of waiting through long
    # fixed sleeps.
    from .capture import is_menu_frame

    menu_deadline = time.monotonic() + 30.0
    while time.monotonic() < menu_deadline:
        frame = _grab_frame(info)
        if frame is None:
            time.sleep(0.2)
            continue
        if is_menu_frame(frame):
            focus(info.hwnd)
            keyboard.tap("SPACE")
            time.sleep(0.25)
            continue
        break
    time.sleep(0.4)
    # The main menu does not write the in-game SaveState, so wait for it to
    # tick before starting the run clock (nudging the menu with SPACE).
    from .savestate import PlayerMemoryV2
    from .verify import vision_hint

    memory = PlayerMemoryV2()
    memory.attach()
    ready_deadline = time.monotonic() + 30.0
    while time.monotonic() < ready_deadline:
        hint = vision_hint(info)
        if memory.ensure_address(*(hint or (None, None)), timeout_s=4.0):
            state = memory.read()
            if state is not None and state.time_stamp > 30:
                break
        focus(info.hwnd)
        keyboard.tap("SPACE")
        time.sleep(1.0)
    else:
        print("warning: game state not live, playing anyway")
    return _run_play(
        cfg,
        info,
        brain=getattr(args, "brain", None),
        minutes=getattr(args, "minutes", None),
        quiet=getattr(args, "quiet", False),
        no_ui=getattr(args, "no_ui", False),
        legacy=getattr(args, "legacy", False),
    )


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    controller = Controller(cfg, info)
    controller.verbose = True
    with GameView(info) as view:
        view.calibrate_viewport()
        first = view.grab()
        tracker = CameraTracker(first.shape[:2])
        for _ in range(20):
            controller.step(view, tracker)
            time.sleep(0.05)
        print(
            f"player=({controller.state.x:.0f},{controller.state.y:.0f}) "
            f"grounded={controller.state.grounded} "
            f"solid={len(controller.collision.solid_hits)} "
            f"pred_solid={len(controller.collision.predicted_solid)} "
            f"free={len(controller.collision.free_hits)}"
        )
        near_pred = sorted(
            (
                (cx * 8, cy * 8)
                for cx, cy in controller.collision.predicted_solid
                if abs(cx * 8 - controller.state.x) < 96 and abs(cy * 8 - controller.state.y) < 96
            ),
            key=lambda p: (p[1], p[0]),
        )
        near_known = sorted(
            (
                (cx * 8, cy * 8)
                for cx, cy in controller.collision.solid_hits
                if abs(cx * 8 - controller.state.x) < 96 and abs(cy * 8 - controller.state.y) < 96
            ),
            key=lambda p: (p[1], p[0]),
        )
        print(f"known solid near player: {near_known}")
        print(f"predicted solid near player: {near_pred}")
        controller._refresh_visual_map(first, tracker.offset_x, tracker.offset_y)
        candidates = controller._candidates()
        print(f"candidates: {len(candidates)}")
        for cand, plan in candidates:
            print(f"  {cand.key} kind={cand.kind} {cand.description}")
            print(
                f"      walk={cand.walk_offset:+.0f} drift={cand.drift} "
                f"landing={plan.landing} known={plan.landing_known}"
            )
        if not candidates:
            takeoffs = controller._reachable_takeoffs()
            print(f"reachable takeoffs: {takeoffs}")
    return 0


def cmd_calibrate_drift(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    info = wait_for_window(cfg.window.process_name, cfg.window.title, timeout_s=5.0)
    charges = [float(c) for c in args.charges.split(",")]
    out = Path(args.out) if args.out else Path(cfg.play.runs_dir) / "calibration" / "drift.json"
    _traces, fit = run_drift_calibration(info, charges, output=out)
    print(f"gravity={fit['gravity']:.1f} px/s^2")
    print(f"air_speed={fit['air_speed']:.1f} px/s")
    print(f"traces_used={fit['samples']:.0f}")
    print(f"saved to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ajk", description="Auto Jump King")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show game window info")
    p_info.set_defaults(func=cmd_info)

    p_focus = sub.add_parser("focus", help="focus the game window")
    p_focus.set_defaults(func=cmd_focus)

    p_shot = sub.add_parser("shot", help="save a single screenshot")
    p_shot.add_argument("--out", type=str, default=None)
    p_shot.add_argument("--scale", type=float, default=1.0)
    p_shot.set_defaults(func=cmd_shot)

    p_watch = sub.add_parser("watch", help="save a stream of screenshots")
    p_watch.add_argument("--duration", type=float, default=2.0)
    p_watch.add_argument("--interval", type=float, default=0.25)
    p_watch.add_argument("--scale", type=float, default=1.0)
    p_watch.set_defaults(func=cmd_watch)

    p_track = sub.add_parser("track", help="live player/camera tracking debug output")
    p_track.add_argument("--duration", type=float, default=5.0)
    p_track.add_argument("--interval", type=float, default=0.05)
    p_track.add_argument("--scale", type=float, default=0.5)
    p_track.add_argument("--game-dir", type=str, default=None)
    p_track.add_argument("--save-frames", action="store_true")
    p_track.set_defaults(func=cmd_track)

    p_cal = sub.add_parser("calibrate-physics", help="measure jump arcs from the live game")
    p_cal.add_argument(
        "--charges",
        type=str,
        default="0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75",
    )
    p_cal.add_argument("--sample-s", type=float, default=2.0)
    p_cal.add_argument("--out", type=str, default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    p_drift = sub.add_parser("calibrate-drift", help="measure jump arcs with a direction held")
    p_drift.add_argument("--charges", type=str, default="0.1,0.25,0.4,0.55,0.65")
    p_drift.add_argument("--out", type=str, default=None)
    p_drift.set_defaults(func=cmd_calibrate_drift)

    p_play = sub.add_parser("play", help="autonomously play the game")
    p_play.add_argument("--brain", choices=["heuristic", "typesafe", "local"], default=None)
    p_play.add_argument("--minutes", type=float, default=None)
    p_play.add_argument("--no-ui", action="store_true")
    p_play.add_argument("--quiet", action="store_true")
    p_play.add_argument("--legacy", action="store_true", help="use the old reactive controller")
    p_play.set_defaults(func=cmd_play)

    p_auto = sub.add_parser("autostart", help="one-click: launch game and play")
    p_auto.add_argument("--minutes", type=float, default=None)
    p_auto.add_argument("--no-ui", action="store_true")
    p_auto.add_argument("--quiet", action="store_true")
    p_auto.set_defaults(func=cmd_autostart)

    p_plan = sub.add_parser("plan", help="debug: show jump candidates from live state")
    p_plan.set_defaults(func=cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
