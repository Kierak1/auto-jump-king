"""Jump planning: takeoff search, platform plans, candidate generation.

Expects these attributes on the controller: collision, state, params,
blocked_takeoffs, recent_failures, screen_cell, verbose.
"""

from __future__ import annotations

from .brain import Candidate
from .physics import JumpPlan, simulate_jump
from .platforms import Platform, extract_platforms

CHARGE_STEPS = (0.2, 0.3, 0.4, 0.45, 0.5, 0.525, 0.55, 0.575, 0.6)


class PlanningMixin:
    def _takeoff_blocked(self, x: float, radius: float = 30.0) -> bool:
        for bx, _when in self.blocked_takeoffs:
            if abs(x - bx) < radius:
                return True
        return False

    def _takeoff_clear(self, x: float, y: float, half_width: float = 16.0) -> bool:
        if self._takeoff_blocked(x):
            return False
        for dy in (-6.0, -14.0, -22.0):
            for dx in (-half_width, half_width):
                if self.collision.is_solid(x + dx, y + dy):
                    return False
        for dy in (-8.0, -18.0):
            for dx in (-24.0, 24.0):
                if self.collision.is_solid(x + dx, y + dy):
                    return False
        return True

    def _reachable_takeoffs(self, max_offset: float = 120.0, step: float = 12.0) -> list[float]:
        x0, y0 = self.state.x, self.state.y
        offsets = [0.0] if self._takeoff_clear(x0, y0) else []
        for sign in (1.0, -1.0):
            offset = step
            while offset <= max_offset:
                target_x = x0 + sign * offset
                if not self._walk_path_clear(x0, target_x, y0):
                    break
                if self._takeoff_clear(target_x, y0):
                    offsets.append(sign * offset)
                offset += step
        return offsets

    def _walk_path_clear(self, x_from: float, x_to: float, y: float) -> bool:
        step = 4.0
        x = x_from
        direction = 1.0 if x_to > x_from else -1.0
        while (x - x_to) * direction < 0:
            x += direction * step
            if not self.collision.is_solid(x, y + 4.0):
                return False
            if self.collision.is_solid(x, y - 16.0):
                return False
        if not self.collision.is_solid(x_to, y + 4.0):
            return False
        return not self.collision.is_solid(x_to, y - 16.0)

    def _all_platforms(self) -> list[Platform]:
        return extract_platforms(self.collision.tile_set) + extract_platforms(
            self.collision.neighbor_tiles
        )

    def _platform_plans(self) -> list[tuple[Candidate, JumpPlan]]:
        if self.collision.atlas is None:
            return []
        platforms = self._all_platforms()
        x0, y0 = self.state.x, self.state.y
        results: list[tuple[Candidate, JumpPlan]] = []
        for index, platform in enumerate(platforms):
            if platform.top_y > y0:
                continue
            if platform.width < 12:
                continue
            height = y0 - platform.top_y
            near_edge = platform.x_start + 6.0 if platform.x_start > x0 else platform.x_end - 6.0
            variants: list[tuple[float, float, int, float, JumpPlan]] = []
            for charge in CHARGE_STEPS:
                v0 = self.params.v0_for_charge(charge)
                gravity = self.params.gravity
                rise = v0 * v0 / (2.0 * gravity)
                if rise < height + 2.0:
                    continue
                apex = y0 - rise
                t_apex = v0 / gravity
                t_fall = (
                    (2.0 * (platform.top_y - apex) / gravity) ** 0.5
                    if apex < platform.top_y
                    else 0.0
                )
                t_flight = t_apex + t_fall
                for direction in (-1, 0, 1):
                    vx = direction * self.params.air_max_speed
                    if direction == 0:
                        if not platform.contains(x0, margin=2.0):
                            continue
                        targets = [x0]
                    else:
                        targets = [(platform.x_start + platform.x_end) / 2.0, near_edge]
                    for target in targets:
                        ideal = target - vx * t_flight
                        offsets = (0.0, -4.0, 4.0, -8.0, 8.0)
                        offsets += tuple(
                            sign * float(d) for d in range(12, 45, 4) for sign in (-1.0, 1.0)
                        )
                        for d_off in offsets:
                            takeoff_x = ideal + d_off
                            if abs(takeoff_x - x0) > 140:
                                continue
                            if not self._walk_path_clear(x0, takeoff_x, y0):
                                continue
                            if not self._takeoff_clear(takeoff_x, y0):
                                continue
                            plan = simulate_jump(
                                takeoff_x,
                                y0,
                                direction,
                                charge,
                                self.params,
                                self.collision,
                            )
                            if plan.outcome != "landed" or plan.landing is None:
                                continue
                            lx, ly = plan.landing
                            if abs(ly - platform.top_y) > 5:
                                continue
                            if not platform.contains(lx, margin=2.0):
                                continue
                            if plan.max_rise < height + 2:
                                continue
                            offset = takeoff_x - x0
                            score = abs(offset) * 0.4 + charge * 25
                            margin_plan = simulate_jump(
                                takeoff_x,
                                y0,
                                direction,
                                charge,
                                self.params,
                                self.collision,
                                player_half_width=22.0,
                            )
                            robust = (
                                margin_plan.outcome == "landed"
                                and margin_plan.landing is not None
                                and abs(margin_plan.landing[1] - platform.top_y) <= 5
                            )
                            if robust:
                                score -= 60.0
                            variants.append((score, offset, direction, charge, plan, robust))
            if not variants:
                continue
            robust_variants = [v for v in variants if v[5]]
            if robust_variants:
                variants = robust_variants
            variants.sort(key=lambda v: v[0])
            for rank, (_, offset, direction, charge, plan, _robust) in enumerate(variants[:3]):
                cand = Candidate(
                    key=f"p{index}" if rank == 0 else f"p{index}v{rank}",
                    direction=direction,
                    charge_s=charge,
                    v0=plan.v0,
                    dx=plan.landing[0] - x0,
                    dy=y0 - plan.landing[1],
                    outcome=plan.outcome,
                    known_landing=plan.landing_known,
                    description="",
                    kind="jump",
                    apex_rise=plan.max_rise,
                    walk_offset=offset,
                    drift="hold",
                )
                walk_text = ""
                if abs(offset) >= 4:
                    walk_text = (
                        f"walk {'right' if offset > 0 else 'left'} {abs(offset):.0f}px then "
                    )
                cand.description = (
                    f"{walk_text}jump {self._direction_name(direction)} "
                    f"(charge {charge:.2f}s) "
                    f"-> lands on platform at y={platform.top_y:.0f} "
                    f"(x {platform.x_start:.0f}..{platform.x_end:.0f}), "
                    f"gaining {cand.dy:.0f}px of height"
                )
                results.append((cand, plan))
        results.sort(key=lambda cp: (-cp[0].dy, abs(cp[0].walk_offset)))
        return results

    def _simulate_all(self) -> list[tuple[Candidate, JumpPlan]]:
        x0, y0 = self.state.x, self.state.y
        charges = list(CHARGE_STEPS)
        results: list[tuple[Candidate, JumpPlan]] = []
        takeoffs = self._reachable_takeoffs()
        for offset in takeoffs:
            for direction in (-1, 0, 1):
                for charge in charges:
                    plan = simulate_jump(
                        x0 + offset,
                        y0,
                        direction,
                        charge,
                        self.params,
                        self.collision,
                    )
                    if plan.outcome != "landed" or plan.landing is None:
                        continue
                    lx, ly = plan.landing
                    dy = y0 - ly
                    dx = lx - x0
                    results.append(
                        (
                            Candidate(
                                key="",
                                direction=direction,
                                charge_s=charge,
                                v0=plan.v0,
                                dx=dx,
                                dy=dy,
                                outcome=plan.outcome,
                                known_landing=plan.landing_known,
                                description="",
                                kind="jump",
                                apex_rise=plan.max_rise,
                                walk_offset=offset,
                            ),
                            plan,
                        )
                    )
        results.extend(self._refine_takeoffs(results, x0, y0))
        return results

    def _refine_takeoffs(
        self,
        coarse: list[tuple[Candidate, JumpPlan]],
        x0: float,
        y0: float,
    ) -> list[tuple[Candidate, JumpPlan]]:
        if not coarse:
            return []
        ranked = sorted(coarse, key=lambda cp: (-cp[0].dy, abs(cp[0].dx)))
        seeds: list[float] = []
        for cand, _ in ranked:
            if all(abs(cand.walk_offset - s) > 12 for s in seeds):
                seeds.append(cand.walk_offset)
            if len(seeds) >= 3:
                break
        refined: list[tuple[Candidate, JumpPlan]] = []
        for seed in seeds:
            for delta in (-8.0, -4.0, 4.0, 8.0):
                offset = seed + delta
                if abs(offset) > 140:
                    continue
                target_x = x0 + offset
                if not self._walk_path_clear(x0, target_x, y0):
                    continue
                if not self._takeoff_clear(target_x, y0):
                    continue
                for direction in (-1, 0, 1):
                    for charge in (0.4, 0.45, 0.5, 0.55, 0.6):
                        plan = simulate_jump(
                            target_x,
                            y0,
                            direction,
                            charge,
                            self.params,
                            self.collision,
                        )
                        if plan.outcome != "landed" or plan.landing is None:
                            continue
                        lx, ly = plan.landing
                        refined.append(
                            (
                                Candidate(
                                    key="",
                                    direction=direction,
                                    charge_s=charge,
                                    v0=plan.v0,
                                    dx=lx - x0,
                                    dy=y0 - ly,
                                    outcome=plan.outcome,
                                    known_landing=plan.landing_known,
                                    description="",
                                    kind="jump",
                                    apex_rise=plan.max_rise,
                                    walk_offset=offset,
                                ),
                                plan,
                            )
                        )
        return refined

    def _candidates(self) -> list[tuple[Candidate, JumpPlan]]:
        platform_plans = [cp for cp in self._platform_plans() if not self._recently_failed(cp[0])]
        if platform_plans:
            return platform_plans[:6]
        all_plans = self._simulate_all()
        platforms = self._all_platforms()
        progress: list[tuple[Candidate, JumpPlan]] = []
        for cand, plan in all_plans:
            if abs(cand.dx) < 2 and cand.dy < 3:
                continue
            if cand.dy <= 1 or plan.landing is None:
                continue
            lx, ly = plan.landing
            if not any(
                p.width >= 12 and abs(p.top_y - ly) <= 4 and p.x_start + 3 <= lx <= p.x_end - 3
                for p in platforms
            ):
                continue
            progress.append((cand, plan))
        progress.sort(key=lambda cp: (-cp[0].dy, abs(cp[0].dx)))
        deduped: list[tuple[Candidate, JumpPlan]] = []
        for cand, plan in progress:
            if any(abs(cand.dx - d.dx) < 8 and abs(cand.dy - d.dy) < 8 for d, _ in deduped):
                continue
            if self._recently_failed(cand):
                continue
            deduped.append((cand, plan))
            if len(deduped) >= 6:
                break
        if deduped:
            for i, (cand, _) in enumerate(deduped):
                cand.key = f"j{i}"
                cand.description = self._describe(cand)
            return deduped
        probes: list[tuple[Candidate, JumpPlan]] = []
        for cand, plan in all_plans:
            if not cand.known_landing:
                continue
            leaves_screen = any(
                p.x < 0 or p.x > 480 or p.y < -360 or p.y > 360 for p in plan.points if p.vy < 0
            )
            if not leaves_screen:
                continue
            cand.kind = "probe"
            probes.append((cand, plan))
        probes.sort(key=lambda cp: (-cp[0].apex_rise, abs(cp[0].dx)))
        selected: list[tuple[Candidate, JumpPlan]] = []
        used_directions: set[int] = set()
        for cand, plan in probes:
            if cand.direction in used_directions:
                continue
            used_directions.add(cand.direction)
            selected.append((cand, plan))
            if len(selected) >= 3:
                break
        for i, (cand, _) in enumerate(selected):
            cand.key = f"p{i}"
            cand.description = (
                f"PROBE: {self._direction_name(cand.direction)}, charge {cand.charge_s:.2f}s, "
                f"rises {cand.apex_rise:.0f}px and crosses the screen edge into "
                f"unmapped space (the next screen up)"
            )
        if selected:
            return selected
        return self._exploration_candidates(all_plans)

    def _exploration_candidates(
        self, all_plans: list[tuple[Candidate, JumpPlan]]
    ) -> list[tuple[Candidate, JumpPlan]]:
        """When nothing gains height, offer the highest jumps in each direction so
        the bot can still try something instead of standing still."""
        best_by_direction: dict[int, tuple[Candidate, JumpPlan]] = {}
        for cand, plan in all_plans:
            current = best_by_direction.get(cand.direction)
            if current is None or cand.apex_rise > current[0].apex_rise:
                best_by_direction[cand.direction] = (cand, plan)
        out: list[tuple[Candidate, JumpPlan]] = []
        for direction in (-1, 0, 1):
            entry = best_by_direction.get(direction)
            if entry is None:
                continue
            cand, plan = entry
            cand.kind = "probe"
            cand.key = f"e{len(out)}"
            cand.description = (
                f"EXPLORE: {self._direction_name(direction)}, charge {cand.charge_s:.2f}s, "
                f"rises {cand.apex_rise:.0f}px and falls back to the same platform"
            )
            out.append((cand, plan))
        return out

    def _recently_failed(self, cand: Candidate) -> bool:
        """Failures persist for the current screen (they are cleared on a camera
        flip), so the bot never retries a jump that the real game bounced."""
        takeoff_x = self.state.x + cand.walk_offset
        for failed_x, charge, direction, _when in self.recent_failures:
            if (
                abs(failed_x - takeoff_x) < 10
                and abs(charge - cand.charge_s) < 0.06
                and int(direction) == cand.direction
            ):
                return True
        if self._takeoff_blocked(takeoff_x):
            return True
        return False

    @staticmethod
    def _direction_name(direction: int) -> str:
        return {1: "right", -1: "left", 0: "straight up"}[direction]

    def _describe(self, cand: Candidate) -> str:
        parts = []
        if abs(cand.walk_offset) >= 10:
            parts.append(
                f"walk {'right' if cand.walk_offset > 0 else 'left'} "
                f"{abs(cand.walk_offset):.0f}px, then"
            )
        parts.append(f"{self._direction_name(cand.direction)}")
        parts.append(f"charge {cand.charge_s:.2f}s,")
        parts.append(f"rises {cand.dy:.0f}px, drifts {cand.dx:+.0f}px,")
        parts.append(f"lands {'on known ground' if cand.known_landing else 'on unmapped ground'}")
        return " ".join(parts)

    def _escape_direction(self) -> float:
        for distance in (24.0, 48.0, 72.0, 96.0):
            for sign in (1.0, -1.0):
                x = self.state.x + sign * distance
                if self._takeoff_clear(x, self.state.y) and not self._takeoff_blocked(x):
                    if self._walk_path_clear(self.state.x, x, self.state.y):
                        return sign * distance
        return 0.0
