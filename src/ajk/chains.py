"""Windowed backward chain planner (the 4->3->2->1 rule).

For a standing king we take the at most `WINDOW_PLATFORMS` platforms directly
above him, then work from the HIGHEST to the lowest: for every platform we ask
the EdgeFactory which takeoffs (on platforms at or below it, inside the
window) provably land on it. A breadth-first search over these incoming edges
finds a chain from the king's current platform to the highest platform that is
connected all the way down to his feet. Only the first jump of the chain is
executed; the window is recomputed after every landing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .edges import MAX_DROP, MAX_RISE, Edge, EdgeFactory
from .gamephysics import WorldGrid
from .planner import Platform, platform_below

WINDOW_PLATFORMS = 30
WINDOW_BELOW = 8
DROP_REACH = 190.0
LEVEL_TOLERANCE = 6.0
SAME_LEVEL_SLACK = 0.5


@dataclass
class Link:
    source: int
    target: int
    edge: Edge


@dataclass
class Chain:
    links: list[Link] = field(default_factory=list)
    robust: bool = True
    tried: int = 0
    revisit: bool = False

    @property
    def first(self) -> Link | None:
        return self.links[0] if self.links else None

    @property
    def terminal(self) -> int | None:
        return self.links[-1].target if self.links else None

    def describe(self, platforms: list[Platform] | None = None) -> str:
        if not self.links:
            return "(empty chain)"
        if platforms is None:
            return f"{len(self.links)} jumps, {'robust' if self.robust else 'best effort'}"
        steps = " -> ".join(f"y={platforms[link.target].top_y:.0f}" for link in self.links)
        return f"{len(self.links)} jumps, {'robust' if self.robust else 'best effort'}: {steps}"


class ChainPlanner:
    def __init__(self, grid: WorldGrid, platforms: list[Platform], factory: EdgeFactory) -> None:
        self.grid = grid
        self.platforms = platforms
        self.factory = factory
        self.index_of = {id(p): i for i, p in enumerate(platforms)}
        self.current = -1
        self.window: list[int] = []
        self.blocked: set[tuple[float, int, int]] = set()
        self.avoid_nodes: set[int] = set()
        self._direct_edges: dict[int, list[Edge]] = {}

    def plan(
        self,
        x: float,
        y: float,
        window_size: int = WINDOW_PLATFORMS,
        avoid: set[int] | None = None,
    ) -> Chain | None:
        feet = y + 26.0
        current = platform_below(self.platforms, x, feet)
        if current is None:
            return None
        self.current = self.index_of[id(current)]
        self.window = self._window(current, feet, window_size)
        ordered = sorted(self.window, key=lambda i: self.platforms[i].top_y)
        self.avoid_nodes = avoid or set()
        avoided = self.avoid_nodes
        chain = self._greedy_chain(avoided)
        if chain is not None:
            return chain
        chain = self._search(ordered, blocked=avoided)
        if chain is not None:
            return chain
        if avoided:
            self.avoid_nodes = set()
            chain = self._search(ordered, blocked=set())
            if chain is not None:
                chain.revisit = True
                return chain
        return Chain(links=[], robust=False, tried=0)

    def _greedy_chain(self, avoided: set[int]) -> Chain | None:
        """Fast path: every direct jump from the current platform was already
        verified while building the window. Take the highest one that strictly
        rises and can continue, if any; otherwise the highest rising one.
        Multi-jump chains (dips, pockets) still go through `_search`."""
        current_y = self.platforms[self.current].top_y
        rising: list[tuple[int, list[Edge]]] = []
        for target, edges in self._direct_edges.items():
            if target in avoided or self.platforms[target].top_y >= current_y - 2.0:
                continue
            usable = [
                edge
                for edge in edges
                if (round(edge.takeoff_x, 1), edge.direction, edge.frames) not in self.blocked
            ]
            if usable:
                rising.append((target, usable))
        if not rising:
            return None
        rising.sort(key=lambda item: self.platforms[item[0]].top_y)
        for target, edges in rising[:3]:
            if self._has_continuation(target):
                edge = next((e for e in edges if e.robust), edges[0])
                return Chain(
                    links=[Link(source=self.current, target=target, edge=edge)],
                    robust=edge.robust,
                    tried=1,
                )
        target, edges = rising[0]
        edge = next((e for e in edges if e.robust), edges[0])
        return Chain(
            links=[Link(source=self.current, target=target, edge=edge)],
            robust=edge.robust,
            tried=1,
        )

    def _search(self, ordered: list[int], blocked: set[int]) -> Chain | None:
        tried = 0
        current_y = self.platforms[self.current].top_y
        continuing: Chain | None = None
        above_dead: Chain | None = None
        below_dead: Chain | None = None
        for target in ordered:
            if target in blocked:
                continue
            tried += 1
            links = self._path(target, robust_only=True)
            if links is None:
                links = self._path(target, robust_only=False)
            if links is None:
                continue
            robust = all(link.edge.robust for link in links)
            chain = Chain(links=links, robust=robust, tried=tried)
            above = self.platforms[target].top_y < current_y - 2.0
            if self._has_continuation(target):
                # A strictly higher target with a continuation is the answer;
                # a lower one (a designed drop) is kept as the best fallback.
                if above:
                    return chain
                if continuing is None:
                    continuing = chain
            elif continuing is None:
                if above and above_dead is None:
                    above_dead = chain
                elif not above and below_dead is None:
                    below_dead = chain
        return continuing or above_dead or below_dead

    def _has_continuation(self, target: int, depth: int = 3) -> bool:
        """True when the target can reach a strictly higher platform through a
        short chain of window jumps (dips allowed), i.e. landing there is not a
        local dead end. The current platform and avoided nodes do not count:
        jumping back where we came from is a loop, not progress."""
        target_y = self.platforms[target].top_y
        # A drop target only counts as continuing when its short chain gets
        # back above the king's current footing, not merely above the drop.
        ceiling = min(target_y, self.platforms[self.current].top_y) - 2.0
        blocked = {self.current} | self.avoid_nodes
        seen = {target}
        frontier = [target]
        for _ in range(depth):
            nxt: list[int] = []
            for node in frontier:
                for other in self.window:
                    if other in seen or other in blocked:
                        continue
                    edges = self.factory.edges_to(other, node)
                    edges = [
                        edge
                        for edge in edges
                        if (round(edge.takeoff_x, 1), edge.direction, edge.frames)
                        not in self.blocked
                    ]
                    if not edges:
                        continue
                    seen.add(other)
                    if self.platforms[other].top_y < ceiling:
                        return True
                    nxt.append(other)
            if not nxt:
                break
            frontier = nxt
        return False

    def _window(self, current: Platform, feet: float, window_size: int) -> list[int]:
        low = current.screen
        high = current.screen + 2
        candidates = [
            i
            for i, p in enumerate(self.platforms)
            if p is not current
            and low <= p.screen <= high
            and p.top_y <= feet + LEVEL_TOLERANCE
            and p.width >= 8.0
        ]
        candidates.sort(key=lambda i: self.platforms[i].top_y, reverse=True)
        above = candidates[:window_size]
        # A staircase of 8px steps can fill the whole "lowest platforms above"
        # budget and hide the real continuation (e.g. a wide platform at the
        # same height but far away horizontally). Add the horizontally nearest
        # candidates too, so directly-reachable platforms stay in the window.
        near = sorted(
            candidates,
            key=lambda i: (
                max(current.x0 - self.platforms[i].x1, self.platforms[i].x0 - current.x1, 0.0),
                -self.platforms[i].top_y,
            ),
        )
        for index in near[:window_size]:
            if index not in above:
                above.append(index)
        below_all = [
            i
            for i, p in enumerate(self.platforms)
            if p is not current
            and low <= p.screen <= high
            and feet + LEVEL_TOLERANCE < p.top_y <= feet + DROP_REACH
            and p.width >= 8.0
        ]
        below_all.sort(key=lambda i: self.platforms[i].top_y)
        # Probe every drop candidate: a designed drop may sit far down the
        # list, behind 8px staircase steps. Reachable ones join the window.
        ordered: list[int] = list(below_all)
        for index in near + candidates:
            if index not in ordered:
                ordered.append(index)
        self._direct_edges = {}
        self._add_reachable(current, ordered, above)
        return list(dict.fromkeys(above + below_all[:WINDOW_BELOW]))

    def _add_reachable(self, current: Platform, candidates: list[int], chosen: list[int]) -> None:
        """Record every platform the king can jump to directly from where he
        stands, and make sure it is part of the window. Staircases of 8px steps
        otherwise consume the whole window budget and hide the one wide
        platform the route actually uses."""
        current_index = self.index_of[id(current)]
        for index in candidates:
            edges = self._direct_edges.get(index)
            if edges is None:
                edges = self.factory.edges_to(index, current_index)
                if edges:
                    self._direct_edges[index] = edges
            if edges and index not in chosen:
                chosen.append(index)

    def _sources(self, target: int) -> list[int]:
        target_y = self.platforms[target].top_y
        out = [
            i
            for i in self.window
            if target_y - MAX_DROP <= self.platforms[i].top_y <= target_y + MAX_RISE
        ]
        current_y = self.platforms[self.current].top_y
        if target_y - MAX_DROP <= current_y <= target_y + MAX_RISE:
            out.insert(0, self.current)
        return out

    def _path(self, target: int, robust_only: bool) -> list[Link] | None:
        prev: dict[int, tuple[int, Edge]] = {}
        seen: set[int] = {target}
        queue: deque[int] = deque([target])
        while queue:
            node = queue.popleft()
            for source in self._sources(node):
                if source in seen:
                    continue
                if source in self.avoid_nodes and source != self.current:
                    continue
                edges = self.factory.edges_to(node, source)
                if robust_only:
                    edges = [edge for edge in edges if edge.robust]
                edges = [
                    edge
                    for edge in edges
                    if (round(edge.takeoff_x, 1), edge.direction, edge.frames) not in self.blocked
                ]
                if not edges:
                    continue
                seen.add(source)
                prev[source] = (node, edges[0])
                if source == self.current:
                    return self._reconstruct(target, prev)
                queue.append(source)
        return None

    def _reconstruct(self, target: int, prev: dict[int, tuple[int, Edge]]) -> list[Link]:
        links: list[Link] = []
        node = self.current
        while node != target:
            nxt, edge = prev[node]
            links.append(Link(source=node, target=nxt, edge=edge))
            node = nxt
        return links
