# Auto Jump King

An autonomous Jump King player. It perceives the game (screen capture + game memory),
plans jumps against the game's real collision map, and lets a
[TypeSafe System One model](https://typesafe.ai) (Jev) pick the next platform to climb.
Precise synthetic keyboard input executes the decision.

```
capture -> vision + memory -> level atlas collision -> platform extraction
        -> jump simulation (candidates) -> Jev Choice -> precise execution
```

## How it plays
- The level's true collision comes from `Content/level.xnb` (13x13 screens, 8px tiles).
- The current screen is identified by matching the frame against the game's own
  midground art, so the correct collision cell is always loaded.
- For every platform in the screen, the planner searches takeoff position, charge,
  direction and air-drift mode with a physics simulation measured from the real game.
- The brain receives a compact state (player, platforms, candidate jumps, rules) and
  picks the jump that lands on the highest platform.

## Setup
```bash
uv sync
cp .env.example .env   # TYPESAFE_API_KEY=... for Jev, or local LLM settings
```

## Usage
```bash
uv run ajk play                       # play with the GUI overlay (default)
uv run ajk play --brain typesafe      # use real Jev (TYPESAFE_API_KEY)
uv run ajk play --brain local         # use a local OpenAI-compatible model
uv run ajk plan                       # show live candidates without playing
uv run ajk calibrate-physics          # re-measure jump physics from the game
uv run ajk shot --out frame.png       # save one frame
```
Press F8 to abort and release all keys. Input is only sent while the game is focused.

## Brain backends
| backend | how | notes |
| --- | --- | --- |
| `heuristic` | built-in planner | default, no API needed |
| `typesafe` | `typesafe-sdk` -> Jev | needs `TYPESAFE_API_KEY` |
| `local` | `system-one-adapter` -> any OpenAI-compatible server | `AJK_LOCAL_BASE_URL`, `AJK_LOCAL_MODEL` |
