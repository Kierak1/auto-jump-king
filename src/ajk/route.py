"""Global route values over the platform graph.

For every platform we compute the best (lowest) world y that is reachable from
it by any chain of jumps. Dead ends keep their own height, so the controller
can prefer jumps whose target leads onward. The computation is cached to disk
(the level is static).
"""

from __future__ import annotations

import time
from pathlib import Path

from .gamephysics import WorldGrid, simulate_jump
from .planner import Platform, platform_below

CACHE_PATH = Path("runs/assets/route_values.npz")


def platform_successors(
    grid: WorldGrid,
    platforms: list[Platform],
    platform: Platform,
    takeoff_step: float = 8.0,
    frames_step: int = 2,
) -> list[Platform]:
    """Platforms reachable by a jump from anywhere on the given platform."""
    reachable: set[Platform] = set()
    y0 = platform.top_y - 26.0
    x = platform.x0 + 1.0
    while x < platform.x1:
        for direction in (-1, 0, 1):
            for frames in range(1, 37, frames_step):
                result = simulate_jump(grid, x, y0, direction, frames)
                if result.outcome != "landed" or result.landing is None:
                    continue
                lx, ly = result.landing
                target = platform_below(platforms, lx, ly)
                if target is not None and target is not platform:
                    reachable.add(target)
        x += takeoff_step
    return list(reachable)


def compute_route_values(
    grid: WorldGrid,
    platforms: list[Platform],
    max_screen: int = 43,
    cache_path: Path = CACHE_PATH,
) -> dict[int, float]:
    """platform id -> best reachable world y (more negative = higher)."""
    if cache_path.exists():
        import numpy as np

        loaded = np.load(cache_path, allow_pickle=True)["values"].item()
        data = {int(k): float(v) for k, v in loaded.items()}
        if len(data) == len(platforms):
            return data
    subset = [p for p in platforms if p.screen <= max_screen]
    index_of = {id(p): i for i, p in enumerate(platforms)}
    id_of = {i: id(p) for i, p in enumerate(platforms)}
    values = {id(p): p.top_y for p in platforms}
    successors: dict[int, list[int]] = {}
    started = time.monotonic()
    for platform in subset:
        i = index_of[id(platform)]
        for target in platform_successors(grid, platforms, platform):
            successors.setdefault(i, []).append(index_of[id(target)])
    print(f"route: {len(subset)} platforms, successors in {time.monotonic() - started:.1f}s")
    changed = True
    passes = 0
    while changed:
        changed = False
        passes += 1
        for i in successors:
            best = values[id_of[i]]
            for j in successors[i]:
                if values[id_of[j]] < best:
                    best = values[id_of[j]]
            if best < values[id_of[i]]:
                values[id_of[i]] = best
                changed = True
    print(f"route: values converged in {passes} passes")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(cache_path, values={str(k): v for k, v in values.items()})
    return values
