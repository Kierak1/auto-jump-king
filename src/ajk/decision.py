"""Model-facing state text and the question/answer round trip.

Expects these attributes on the controller: state, stats, collision, brain,
cfg, verbose, logger, last_state_text, decision_latency, last_decision.
"""

from __future__ import annotations

import time

from .brain import (
    BrainDecision,
    Candidate,
    ChoiceQuestion,
    DecisionContext,
    NoulQuestion,
)
from .physics import JumpPlan


class DecisionMixin:
    def _state_text(self, candidates: list[tuple[Candidate, JumpPlan]]) -> str:
        lines = [
            "Jump King (2D precision platformer). The King climbs a tower; the goal is up.",
            "World coordinates in screen pixels: x grows right, y grows down (smaller y = higher).",
            "",
            "GAME RULES (verified in-game):",
            "- The King can only jump while standing on the ground (no mid-air jump).",
            "- Hold SPACE to charge; release to jump. Longer charge = higher jump.",
            "- The horizontal speed is LOCKED AT TAKEOFF: if LEFT/RIGHT is held when",
            "  the jump starts, the King drifts that way at ~205 px/s for the whole",
            "  flight; a straight-up jump never moves sideways. The direction cannot",
            "  be changed mid-air. A briefly charged jump is low and long; a full jump",
            "  is high and steep.",
            "- Platforms are solid from every side. The King must land ON TOP of a platform;",
            "  he cannot pass through it. Hitting a platform's side makes him BOUNCE OFF",
            "  (his horizontal direction reverses), which ruins the jump.",
            "- Therefore takeoffs must be AWAY from walls: standing next to a wall blocks",
            "  the jump or bounces the King back. Prefer takeoffs near the middle of the",
            "  current platform, then drift in the air.",
            "- The King cannot jump while touching a wall or a prop (campfire, NPC).",
            "  If a jump does not lift off, the spot is blocked - move elsewhere first.",
            "",
            "Every candidate below was simulated against the game's real collision map,",
            "so a candidate that says it lands on a platform clears the walls and lands on top.",
            f"Player: x={self.state.x:.0f} y={self.state.y:.0f} "
            f"vx={self.state.vx:+.0f} vy={self.state.vy:+.0f} "
            f"grounded={'yes' if self.state.grounded else 'no'}.",
        ]
        if self.collision.atlas is not None:
            platforms = self._all_platforms()
            reachable = [p for p in platforms if p.width >= 12]
            if reachable:
                lines.append("Platforms in this screen (top_y, x range, height vs player):")
                for platform in sorted(reachable, key=lambda p: p.top_y):
                    gain = self.state.y - platform.top_y
                    lines.append(
                        f"  - top y={platform.top_y:.0f}, x={platform.x_start:.0f}"
                        f"..{platform.x_end:.0f}, "
                        f"{'ABOVE player by' if gain > 0 else 'below player by'} "
                        f"{abs(gain):.0f}px"
                    )
        if self.stats.falls:
            lines.append(f"Falls so far: {self.stats.falls}.")
        if candidates:
            lines.append("Candidate jumps (precomputed with measured physics):")
            for cand, _ in candidates:
                lines.append(f"  {cand.key}: {cand.description}.")
        else:
            lines.append("No candidate jump currently lands on solid ground.")
        lines.append(
            "Rule: always pick the candidate that lands on the highest reachable "
            "platform. If two candidates reach the same height, pick the one that "
            "requires the least walking. Never pick a candidate that only jumps "
            "straight up unless it lands on a higher platform."
        )
        return "\n".join(lines)

    def _ask_brain(self, candidates: list[tuple[Candidate, JumpPlan]]) -> BrainDecision:
        context = DecisionContext(
            player_x=self.state.x,
            player_y=self.state.y,
            vx=self.state.vx,
            vy=self.state.vy,
            grounded=self.state.grounded,
            candidates=[c for c, _ in candidates],
        )
        state_text = self._state_text(candidates)
        self.last_state_text = state_text
        if self.verbose:
            print("--- state sent to model ---")
            for line in state_text.split("\n"):
                print(f"  {line}")
            print("---------------------------")
        criteria = {c.key: c.description for c, _ in candidates}
        criteria["wait"] = "stay put and re-evaluate"
        questions = {
            "jump": ChoiceQuestion(
                instructions=(
                    "Which jump should the King perform next? Pick the option that "
                    "lands on the highest platform the King can reach. If several "
                    "reach the same platform, prefer the one with the least walking. "
                    "Only pick 'wait' if no jump gains height."
                ),
                criteria=criteria,
            ),
            "stuck": NoulQuestion(
                instructions="The King has been unable to gain height for a while."
            ),
        }
        started = time.perf_counter()
        decision = self.brain.decide(state_text, questions, context)
        self.decision_latency = time.perf_counter() - started
        self.last_decision = decision
        return decision
