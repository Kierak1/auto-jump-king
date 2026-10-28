from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.toml"


@dataclass
class WindowConfig:
    process_name: str = "JumpKing.exe"
    title: str = "Jump King"
    native_width: int = 360
    native_height: int = 640


@dataclass
class InputConfig:
    left_key: str = "LEFT"
    right_key: str = "RIGHT"
    jump_key: str = "SPACE"
    confirm_key: str = "ENTER"
    focus_timeout_s: float = 2.0
    tap_ms: int = 60


@dataclass
class PhysicsConfig:
    gravity: float = 886.4
    air_max_speed: float = 205.0
    max_charge_s: float = 0.62
    terminal_speed: float = 700.0
    charge_v0: list[list[float]] = field(
        default_factory=lambda: [
            [0.1, 69.8],
            [0.2, 197.0],
            [0.3, 262.0],
            [0.35, 326.0],
            [0.4, 390.0],
            [0.5, 453.0],
            [0.55, 490.0],
            [0.6, 524.0],
        ]
    )


@dataclass
class VisionConfig:
    surface_scan: bool = False
    surface_brightness: float = 200.0
    surface_below_max: float = 180.0
    surface_min_run: int = 24
    surface_confirm_frames: int = 2
    use_classifier: bool = False
    screen_check_interval_s: float = 1.0
    tile_min_brightness: int = 60
    tile_edge_threshold: int = 40


@dataclass
class BrainConfig:
    backend: str = "heuristic"
    api_key: str = ""
    model: str = "jev-latest"
    min_confidence: float = 0.5
    timeout_s: float = 10.0
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen3:8b"
    local_api_key: str = "local"


@dataclass
class PlayConfig:
    fps: int = 30
    max_minutes: float = 30.0
    kill_key: str = "F8"
    save_frames: bool = False
    runs_dir: str = "runs"


@dataclass
class Config:
    game_dir: str = r"C:\Games\Jump King"
    window: WindowConfig = field(default_factory=WindowConfig)
    input: InputConfig = field(default_factory=InputConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    play: PlayConfig = field(default_factory=PlayConfig)


def _apply(section: Any, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in data.items():
        if key in valid:
            setattr(section, key, value)


def _apply_env(cfg: Config) -> None:
    mapping = {
        "TYPESAFE_API_KEY": ("brain", "api_key"),
        "AJK_BRAIN": ("brain", "backend"),
        "AJK_MODEL": ("brain", "model"),
        "AJK_LOCAL_BASE_URL": ("brain", "local_base_url"),
        "AJK_LOCAL_MODEL": ("brain", "local_model"),
        "AJK_LOCAL_API_KEY": ("brain", "local_api_key"),
        "AJK_MAX_MINUTES": ("play", "max_minutes"),
        "AJK_KILL_KEY": ("play", "kill_key"),
        "AJK_GAME_DIR": ("", "game_dir"),
    }
    for env_name, (section_name, attr) in mapping.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        section = getattr(cfg, section_name) if section_name else cfg
        current = getattr(section, attr, None)
        if isinstance(current, float):
            raw = float(raw)
        elif isinstance(current, int):
            raw = int(raw)
        setattr(section, attr, raw)


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    cfg_path = path or DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with cfg_path.open("rb") as handle:
            data = tomllib.load(handle)
        for section in fields(cfg):
            section_data = data.get(section.name)
            if isinstance(section_data, dict):
                _apply(getattr(cfg, section.name), section_data)
            elif section_data is not None:
                setattr(cfg, section.name, section_data)
    _apply_env(cfg)
    return cfg
