from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import BrainConfig


@dataclass
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str]


@dataclass
class ScoreQuestion:
    instructions: str
    criteria: list[str]


@dataclass
class NoulQuestion:
    instructions: str


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


@dataclass
class Answer:
    type: str
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class Candidate:
    key: str
    direction: int
    charge_s: float
    v0: float
    dx: float
    dy: float
    outcome: str
    known_landing: bool
    description: str
    kind: str = "jump"
    apex_rise: float = 0.0
    walk_offset: float = 0.0
    drift: str = "hold"


@dataclass
class DecisionContext:
    player_x: float
    player_y: float
    vx: float
    vy: float
    grounded: bool
    candidates: list[Candidate]
    stuck: bool = False


@dataclass
class BrainDecision:
    choice: str
    confidence: float
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)


class Brain(Protocol):
    def decide(
        self,
        state: str,
        questions: dict[str, Question],
        context: DecisionContext,
    ) -> BrainDecision: ...


def _to_sdk(question: Question) -> Any:
    from typesafe_sdk import Choice, Noul, Score

    if isinstance(question, ChoiceQuestion):
        return Choice(instructions=question.instructions, criteria=question.criteria)
    if isinstance(question, ScoreQuestion):
        return Score(instructions=question.instructions, criteria=question.criteria)
    return Noul(instructions=question.instructions)


def _from_sdk(answer: Any) -> Answer:
    kind = getattr(answer, "type", None) or type(answer).__name__.lower()
    probabilities = {}
    raw_probs = getattr(answer, "probabilities", None)
    if raw_probs is not None:
        try:
            probabilities = {str(k): float(v) for k, v in dict(raw_probs).items()}
        except (TypeError, ValueError):
            probabilities = {}
    return Answer(
        type=str(kind),
        choice=getattr(answer, "choice", None),
        score=getattr(answer, "score", None),
        noul=getattr(answer, "noul", None),
        confidence=getattr(answer, "confidence", None),
        probabilities=probabilities,
    )


class TypeSafeBrain:
    def __init__(self, cfg: BrainConfig) -> None:
        from typesafe_sdk import TypeSafeClient

        kwargs: dict[str, Any] = {}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        self.client = TypeSafeClient(**kwargs)
        self.model = cfg.model

    def decide(
        self,
        state: str,
        questions: dict[str, Question],
        context: DecisionContext,
    ) -> BrainDecision:
        sdk_questions = {name: _to_sdk(q) for name, q in questions.items()}
        response = self.client.system_one(
            state=state,
            questions=sdk_questions,
            model=self.model,
        )
        answers = {name: _from_sdk(a) for name, a in response.answers.items()}
        choice_answer = answers.get("jump")
        choice = choice_answer.choice if choice_answer else ""
        confidence = float(choice_answer.confidence or 0.0) if choice_answer else 0.0
        return BrainDecision(
            choice=choice or "",
            confidence=confidence,
            backend="typesafe",
            raw={k: v.__dict__ for k, v in answers.items()},
        )


class LocalLLMBrain:
    def __init__(self, cfg: BrainConfig) -> None:
        from system_one_adapter import SystemOneAdapterClient
        from system_one_adapter.providers.openai import OpenAIProvider

        self.client = SystemOneAdapterClient()
        self.provider = OpenAIProvider(
            cfg.local_model,
            base_url=cfg.local_base_url,
            api_key=cfg.local_api_key,
        )

    def decide(
        self,
        state: str,
        questions: dict[str, Question],
        context: DecisionContext,
    ) -> BrainDecision:
        sdk_questions = {name: _to_sdk(q) for name, q in questions.items()}
        response = self.client.system_one(state, sdk_questions, model=self.provider)
        answers = {name: _from_sdk(a) for name, a in response.answers.items()}
        choice_answer = answers.get("jump")
        choice = choice_answer.choice if choice_answer else ""
        confidence = float(choice_answer.confidence or 0.0) if choice_answer else 0.0
        return BrainDecision(
            choice=choice or "",
            confidence=confidence,
            backend="local",
            raw={k: v.__dict__ for k, v in answers.items()},
        )


class HeuristicBrain:
    def decide(
        self,
        state: str,
        questions: dict[str, Question],
        context: DecisionContext,
    ) -> BrainDecision:
        _ = (state, questions)
        if not context.candidates:
            return BrainDecision(choice="wait", confidence=1.0, backend="heuristic")
        jumps = [c for c in context.candidates if c.kind == "jump" and c.dy > 0]
        if jumps:
            best = max(jumps, key=lambda c: (c.known_landing, c.dy, -abs(c.dx)))
            confidence = 0.9 if best.known_landing else 0.6
        else:
            probes = [c for c in context.candidates if c.kind == "probe"]
            pool = probes or context.candidates
            best = max(pool, key=lambda c: c.apex_rise - 0.2 * abs(c.dx))
            confidence = 0.3
        return BrainDecision(
            choice=best.key,
            confidence=confidence,
            backend="heuristic",
            raw={"dy": best.dy, "dx": best.dx, "outcome": best.outcome, "kind": best.kind},
        )


def make_brain(cfg: BrainConfig) -> Brain:
    if cfg.backend == "typesafe":
        return TypeSafeBrain(cfg)
    if cfg.backend == "local":
        return LocalLLMBrain(cfg)
    return HeuristicBrain()
