from __future__ import annotations

from dataclasses import dataclass

TILE = 8


@dataclass(frozen=True)
class Platform:
    top_y: float
    x_start: float
    x_end: float

    @property
    def width(self) -> float:
        return self.x_end - self.x_start

    def contains(self, x: float, margin: float = 4.0) -> bool:
        return self.x_start - margin <= x <= self.x_end + margin

    def describe(self) -> str:
        return f"top y={self.top_y:.0f}, x={self.x_start:.0f}..{self.x_end:.0f}"


def extract_platforms(tiles: set[tuple[int, int]], min_width: int = 2) -> list[Platform]:
    """List every walkable surface: a solid tile whose tile above is free, grouped
    into horizontal runs. Stacked platforms are all reported (unlike a per-column
    topmost scan)."""
    if not tiles:
        return []
    surfaces: dict[int, list[int]] = {}
    for tx, ty in tiles:
        if (tx, ty - 1) in tiles:
            continue
        surfaces.setdefault(ty, []).append(tx)
    platforms: list[Platform] = []
    for ty, txs in sorted(surfaces.items()):
        txs.sort()
        run_start = prev = txs[0]
        for tx in txs[1:]:
            if tx == prev + 1:
                prev = tx
                continue
            if prev - run_start + 1 >= min_width:
                platforms.append(
                    Platform(float(ty * TILE), float(run_start * TILE), float((prev + 1) * TILE))
                )
            run_start = prev = tx
        if prev - run_start + 1 >= min_width:
            platforms.append(
                Platform(float(ty * TILE), float(run_start * TILE), float((prev + 1) * TILE))
            )
    return platforms
